/-
# Finite datasets, average losses and imbalance factors

Formalizes the data-level identities used throughout the paper:

* Eq. (loss_3)/(loss_4): the average loss over a disjoint union `D_H ∪ D_T` is the
  `γ`-weighted combination of the Head and Tail average losses, `γ = |D_H| / |D|`.
* Eq. (gamma_if)/(one_minus_gamma_if): under Assumption 3.2 (equal class sizes inside the Head
  and inside the Tail), `γ = k_H IF / (k_H IF + k_T)` and `1 - γ = k_T / (k_H IF + k_T)`.
* Eq. (eq:full_decomposition): with a shared regularizer `R`,
  `F_D = (1 - ε) F_{D¹} + ε F_{D_T}`.
* Eq. (eq:epsilon_bound): for partitions sorted by cardinality,
  `ε = |D_T| / |D| ≤ (n - 1) |D²| / |D¹|`.
* Proposition 3.9 (objective-level LTR ↔ CL reformulation).
-/
import Mathlib

open Finset

namespace CLTR

variable {ι Θ : Type*}

/-- Average of the per-sample losses `ℓ i θ` over a finite set of samples `s`:
`𝓛(s, θ) = (1/|s|) ∑_{i ∈ s} ℓ(i, θ)`. -/
noncomputable def avgLoss (ℓ : ι → Θ → ℝ) (s : Finset ι) (θ : Θ) : ℝ :=
  (∑ i ∈ s, ℓ i θ) / s.card

/-- Eq. (loss_3): average loss over a disjoint union is the cardinality-weighted combination. -/
theorem avgLoss_union [DecidableEq ι] (ℓ : ι → Θ → ℝ) {H T : Finset ι} (hdisj : Disjoint H T)
    (hH : H.Nonempty) (hT : T.Nonempty) (θ : Θ) :
    avgLoss ℓ (H ∪ T) θ =
      ((H.card : ℝ) / (H ∪ T).card) * avgLoss ℓ H θ +
      ((T.card : ℝ) / (H ∪ T).card) * avgLoss ℓ T θ := by
  have hHc : (H.card : ℝ) ≠ 0 := by exact_mod_cast hH.card_pos.ne'
  have hTc : (T.card : ℝ) ≠ 0 := by exact_mod_cast hT.card_pos.ne'
  unfold avgLoss
  rw [sum_union hdisj, card_union_of_disjoint hdisj]
  push_cast
  field_simp

/-- Eq. (loss_4): with `γ = |D_H|/|D|`, `𝓛(D) = γ 𝓛(D_H) + (1-γ) 𝓛(D_T)`. -/
theorem avgLoss_union_gamma [DecidableEq ι] (ℓ : ι → Θ → ℝ) {H T : Finset ι}
    (hdisj : Disjoint H T) (hH : H.Nonempty) (hT : T.Nonempty) (θ : Θ) :
    avgLoss ℓ (H ∪ T) θ =
      ((H.card : ℝ) / (H ∪ T).card) * avgLoss ℓ H θ +
      (1 - (H.card : ℝ) / (H ∪ T).card) * avgLoss ℓ T θ := by
  rw [avgLoss_union ℓ hdisj hH hT θ]
  have hU : ((H ∪ T).card : ℝ) ≠ 0 := by
    exact_mod_cast (hH.mono subset_union_left).card_pos.ne'
  congr 2
  rw [card_union_of_disjoint hdisj]
  push_cast
  field_simp
  ring

/-- Under Assumption 3.2, the Head (resp. Tail) has `k_H` (resp. `k_T`) classes of `n_H`
(resp. `n_T`) samples each. This counts the samples of a union of equal-size classes. -/
theorem card_filter_mem_of_equal_class_sizes {κ : Type*} [DecidableEq κ]
    (D : Finset ι) (y : ι → κ) (C : Finset κ) (n : ℕ)
    (hsize : ∀ c ∈ C, (D.filter (fun i => y i = c)).card = n) :
    (D.filter (fun i => y i ∈ C)).card = C.card * n := by
  rw [card_eq_sum_card_fiberwise (f := y) (s := D.filter (fun i => y i ∈ C)) (t := C)
    (fun i hi => (mem_filter.1 hi).2)]
  rw [sum_congr rfl (g := fun _ => n), sum_const, smul_eq_mul]
  intro c hc
  rw [filter_filter, ← hsize c hc]
  congr 1
  apply filter_congr
  intro i _
  constructor
  · rintro ⟨_, h⟩; exact h
  · intro h; exact ⟨h ▸ hc, h⟩

/-- Eq. (gamma_if): `γ = k_H n_H / (k_H n_H + k_T n_T) = k_H IF / (k_H IF + k_T)` with
`IF = n_H / n_T`. -/
theorem gamma_eq_IF (kH kT nH nT : ℝ) (hnT : 0 < nT) :
    kH * nH / (kH * nH + kT * nT) = kH * (nH / nT) / (kH * (nH / nT) + kT) := by
  have : kH * (nH / nT) + kT = (kH * nH + kT * nT) / nT := by field_simp
  rw [this, div_div_eq_mul_div, mul_div_assoc', div_mul_cancel₀ _ hnT.ne']

/-- Eq. (one_minus_gamma_if): `1 - γ = k_T / (k_H IF + k_T)`. -/
theorem one_sub_gamma_eq (kH kT IF : ℝ) (hpos : 0 < kH * IF + kT) :
    1 - kH * IF / (kH * IF + kT) = kT / (kH * IF + kT) := by
  field_simp
  ring

/-- `1 - γ = O(1/IF)`: explicitly `k_T/(k_H IF + k_T) ≤ (k_T/k_H) · IF⁻¹`. -/
theorem one_sub_gamma_le (kH kT IF : ℝ) (hkH : 0 < kH) (hkT : 0 ≤ kT) (hIF : 0 < IF) :
    kT / (kH * IF + kT) ≤ (kT / kH) * IF⁻¹ := by
  rw [div_mul_eq_mul_div, ← div_eq_mul_inv, div_div]
  apply div_le_div_of_nonneg_left hkT (by positivity)
  linarith

/-- Eq. (eq:full_decomposition): with `F_A(θ) = 𝓛(A, θ) + R(θ)` and `ε = |D_T|/|D|`,
`F_D = (1 - ε) F_{D¹} + ε F_{D_T}`, equivalently `F_D = F_{D¹} + ε (F_{D_T} - F_{D¹})`. -/
theorem objective_union [DecidableEq ι] (ℓ : ι → Θ → ℝ) (R : Θ → ℝ) {H T : Finset ι}
    (hdisj : Disjoint H T) (hH : H.Nonempty) (hT : T.Nonempty) (θ : Θ) :
    avgLoss ℓ (H ∪ T) θ + R θ =
      (avgLoss ℓ H θ + R θ) +
        ((T.card : ℝ) / (H ∪ T).card) * ((avgLoss ℓ T θ + R θ) - (avgLoss ℓ H θ + R θ)) := by
  rw [avgLoss_union ℓ hdisj hH hT θ]
  have hU : ((H ∪ T).card : ℝ) ≠ 0 := by
    exact_mod_cast (hH.mono subset_union_left).card_pos.ne'
  have hsum : (H.card : ℝ) / (H ∪ T).card = 1 - (T.card : ℝ) / (H ∪ T).card := by
    rw [card_union_of_disjoint hdisj]
    push_cast
    field_simp
    ring
  rw [hsum]
  ring

/-- Eq. (eq:epsilon_bound): if the partition sizes `s 0 ≥ s 1 ≥ ⋯ ≥ s (n-1) ≥ 0` are sorted,
the relative tail mass satisfies `∑_{i≥1} s i / ∑_i s i ≤ (n - 1) s 1 / s 0`. -/
theorem tail_mass_le (s : ℕ → ℝ) (n : ℕ) (hn : 2 ≤ n) (hnonneg : ∀ i, 0 ≤ s i)
    (hsorted : ∀ i j, i ≤ j → s j ≤ s i) (h0 : 0 < s 0) :
    (∑ i ∈ Ico 1 n, s i) / (∑ i ∈ range n, s i) ≤ (n - 1 : ℝ) * s 1 / s 0 := by
  have hnum : ∑ i ∈ Ico 1 n, s i ≤ (n - 1 : ℝ) * s 1 := by
    calc ∑ i ∈ Ico 1 n, s i ≤ ∑ _i ∈ Ico 1 n, s 1 :=
          sum_le_sum fun i hi => hsorted 1 i (mem_Ico.1 hi).1
      _ = (n - 1 : ℝ) * s 1 := by
          rw [sum_const, Nat.card_Ico, nsmul_eq_mul]
          push_cast [Nat.cast_sub (by omega : 1 ≤ n)]
          ring
  have hden : s 0 ≤ ∑ i ∈ range n, s i :=
    single_le_sum (fun i _ => hnonneg i) (mem_range.2 (by omega))
  have hnumnn : 0 ≤ ∑ i ∈ Ico 1 n, s i := sum_nonneg fun i _ => hnonneg i
  calc (∑ i ∈ Ico 1 n, s i) / (∑ i ∈ range n, s i)
      ≤ (∑ i ∈ Ico 1 n, s i) / s 0 := div_le_div_of_nonneg_left hnumnn h0 hden
    _ ≤ (n - 1 : ℝ) * s 1 / s 0 := div_le_div_of_nonneg_right hnum h0.le

/-- Proposition 3.9 (objective-level reformulation). If the dataset is the union of the
partitions `D = ⋃_t D^t`, then a predictor `f` fits every sample of the LTR dataset iff it fits
every sample of every dataset of the continual-learning stream `D_{𝒴_t} = D^t`. -/
theorem prop3_9_ltr_eq_cl_objective {X Y : Type*} {T : Type*}
    (D : Set (X × Y)) (stream : T → Set (X × Y)) (hD : D = ⋃ t, stream t) (f : X → Y) :
    (∀ p ∈ D, f p.1 = p.2) ↔ (∀ t, ∀ p ∈ stream t, f p.1 = p.2) := by
  subst hD
  simp only [Set.mem_iUnion, forall_exists_index]
  exact ⟨fun h t p hp => h p t hp, fun h p t hp => h t p hp⟩

end CLTR
