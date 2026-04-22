from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def _stable_seed(*items: object) -> int:
    payload = "|".join(str(item) for item in items).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class CountSketch:
    def __init__(self, num_hash: int = 4, sketch_size: int = 256, seed: int = 1234) -> None:
        self.num_hash = num_hash
        self.sketch_size = sketch_size
        self.seed = seed
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _prepare(self, dimension: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        if dimension not in self._cache:
            bucket_indices = []
            signs = []
            for row in range(self.num_hash):
                row_buckets = []
                row_signs = []
                for index in range(dimension):
                    seed = _stable_seed(self.seed, row, index)
                    row_buckets.append(seed % self.sketch_size)
                    row_signs.append(1.0 if ((seed >> 1) & 1) == 0 else -1.0)
                bucket_indices.append(row_buckets)
                signs.append(row_signs)
            self._cache[dimension] = (
                torch.tensor(bucket_indices, dtype=torch.long),
                torch.tensor(signs, dtype=torch.float32),
            )
        bucket_tensor, sign_tensor = self._cache[dimension]
        return bucket_tensor.to(device), sign_tensor.to(device)

    def sketch(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.reshape(-1)
        bucket_indices, signs = self._prepare(vector.numel(), vector.device)
        sketch = torch.zeros(self.num_hash, self.sketch_size, device=vector.device, dtype=vector.dtype)
        sketch.scatter_add_(1, bucket_indices, vector.unsqueeze(0) * signs)
        return sketch

    def recover(self, sketch: torch.Tensor, dimension: int) -> torch.Tensor:
        bucket_indices, signs = self._prepare(dimension, sketch.device)
        estimates = []
        for row in range(self.num_hash):
            estimates.append(sketch[row, bucket_indices[row]] * signs[row])
        return torch.stack(estimates, dim=0).median(dim=0).values


def quantize_tensor(tensor: torch.Tensor, scale_exponent: int) -> torch.Tensor:
    return torch.round(tensor * float(10 ** scale_exponent)).to(dtype=torch.long)


def dequantize_tensor(tensor: torch.Tensor, scale_exponent: int) -> torch.Tensor:
    return tensor.to(dtype=torch.float32) / float(10 ** scale_exponent)


class PairwiseMasking:
    def __init__(self, master_seed: int = 2026, mask_bound: int = 16) -> None:
        self.master_seed = master_seed
        self.mask_bound = mask_bound

    def generate_mask(
        self,
        client_id: int,
        participant_ids: Sequence[int],
        round_idx: int,
        shape: Sequence[int],
        device: torch.device | str,
    ) -> torch.Tensor:
        mask = torch.zeros(*shape, dtype=torch.long, device=device)
        for peer_id in participant_ids:
            if peer_id == client_id:
                continue
            seed = _stable_seed(self.master_seed, round_idx, min(client_id, peer_id), max(client_id, peer_id))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            pair_mask = torch.randint(-self.mask_bound, self.mask_bound + 1, tuple(shape), generator=generator, dtype=torch.long).to(device)
            mask = mask + pair_mask if client_id < peer_id else mask - pair_mask
        return mask


@dataclass
class PaillierPublicKey:
    n: int
    g: int
    n_sq: int


@dataclass
class PaillierPrivateKey:
    lam: int
    mu: int
    public_key: PaillierPublicKey


class PaillierAHE:
    def __init__(self, key_bits: int = 256) -> None:
        self.public_key, self.private_key = self.generate_keypair(key_bits)

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    @staticmethod
    def _is_probable_prime(candidate: int, rounds: int = 16) -> bool:
        if candidate < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if candidate in small_primes:
            return True
        for prime in small_primes:
            if candidate % prime == 0:
                return False
        d = candidate - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(rounds):
            a = secrets.randbelow(candidate - 3) + 2
            x = pow(a, d, candidate)
            if x in (1, candidate - 1):
                continue
            for _ in range(s - 1):
                x = pow(x, 2, candidate)
                if x == candidate - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _generate_prime(cls, bits: int) -> int:
        while True:
            candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
            if cls._is_probable_prime(candidate):
                return candidate

    @classmethod
    def generate_keypair(cls, key_bits: int) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
        p = cls._generate_prime(key_bits // 2)
        q = cls._generate_prime(key_bits // 2)
        while q == p:
            q = cls._generate_prime(key_bits // 2)
        n = p * q
        n_sq = n * n
        g = n + 1
        lam = cls._lcm(p - 1, q - 1)
        l_value = (pow(g, lam, n_sq) - 1) // n
        mu = pow(l_value, -1, n)
        public_key = PaillierPublicKey(n=n, g=g, n_sq=n_sq)
        private_key = PaillierPrivateKey(lam=lam, mu=mu, public_key=public_key)
        return public_key, private_key

    def encrypt_value(self, value: int) -> int:
        plaintext = value % self.public_key.n
        while True:
            r = secrets.randbelow(self.public_key.n - 1) + 1
            if math.gcd(r, self.public_key.n) == 1:
                break
        return (pow(self.public_key.g, plaintext, self.public_key.n_sq) * pow(r, self.public_key.n, self.public_key.n_sq)) % self.public_key.n_sq

    def decrypt_value(self, ciphertext: int) -> int:
        x = pow(ciphertext, self.private_key.lam, self.public_key.n_sq)
        l_value = (x - 1) // self.public_key.n
        plaintext = (l_value * self.private_key.mu) % self.public_key.n
        if plaintext > self.public_key.n // 2:
            plaintext -= self.public_key.n
        return plaintext

    def encrypt_tensor(self, tensor: torch.Tensor) -> list[list[int]]:
        rows = tensor.detach().cpu().to(dtype=torch.long).tolist()
        return [[self.encrypt_value(int(value)) for value in row] for row in rows]

    def decrypt_tensor(self, ciphertext_matrix: Sequence[Sequence[int]], device: torch.device | str) -> torch.Tensor:
        rows = [[self.decrypt_value(int(value)) for value in row] for row in ciphertext_matrix]
        return torch.tensor(rows, dtype=torch.long, device=device)

    def aggregate_ciphertexts(self, ciphertext_matrices: Iterable[Sequence[Sequence[int]]]) -> list[list[int]]:
        ciphertext_matrices = list(ciphertext_matrices)
        if not ciphertext_matrices:
            raise ValueError("No ciphertext matrices provided for aggregation")
        row_count = len(ciphertext_matrices[0])
        column_count = len(ciphertext_matrices[0][0])
        aggregated = [[1 for _ in range(column_count)] for _ in range(row_count)]
        for matrix in ciphertext_matrices:
            for row_idx in range(row_count):
                for col_idx in range(column_count):
                    aggregated[row_idx][col_idx] = (aggregated[row_idx][col_idx] * int(matrix[row_idx][col_idx])) % self.public_key.n_sq
        return aggregated
