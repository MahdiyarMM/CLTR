/-
# Nonconvex case: Theorem 3.5 (KL / Hölder-1/2) and Theorem 3.7 (multiple partitions)

The KL property with exponent `1/2` is used in the paper only through the local Hölder-`1/2`
error bound (Eqs. (holder), (eq:head_error_bound)):

  `dist(θ, S_H) ≤ C √(F_H(θ) - F_H*)`   for all `θ ∈ U`,

where `S_H` is the set of Head minimizers and `U` is a neighborhood of `S_H`. Deriving this error
bound from the KL inequality for definable deep networks is a deep external result (cited in the
paper); here it is taken as an explicit hypothesis, exactly as in the statement of Theorem 3.7
("under the local error-bound and bounded loss-contrast conditions").
-/
import Mathlib
import Proofs.Data

open Set Filter Asymptotics Topology Metric Finset

namespace CLTR

section Perturbation

variable {Θ : Type*}

/-- Eq. (eq:head_loss_bound): if `F = F_H + ε Δ`, `θ*` is no worse than `θ̄` for `F` and the
contrast `Δ` is bounded by `M` at both points, then `F_H(θ*) - F_H(θ̄) ≤ 2 M ε`. -/
theorem head_excess_le {FH F Δ : Θ → ℝ} {ε M : ℝ} (hε : 0 ≤ ε)
    (hF : ∀ θ, F θ = FH θ + ε * Δ θ) {θstar θbar : Θ} (hopt : F θstar ≤ F θbar)
    (hΔs : |Δ θstar| ≤ M) (hΔb : |Δ θbar| ≤ M) :
    FH θstar - FH θbar ≤ 2 * M * ε := by
  rw [hF, hF] at hopt
  have h1 := (abs_le.1 hΔs).1
  have h2 := (abs_le.1 hΔb).2
  nlinarith

/-- Generic perturbation bound behind Theorems 3.5 and 3.7 and the exponential-decay remark:
the Hölder error bound turns the `2Mε` excess Head loss into `dist(θ*, S_H) ≤ C √(2 M ε)`. -/
theorem dist_le_of_perturbation [PseudoMetricSpace Θ] {FH F Δ : Θ → ℝ} {ε M C : ℝ}
    {S U : Set Θ} (hε : 0 ≤ ε) (hC : 0 ≤ C)
    (hF : ∀ θ, F θ = FH θ + ε * Δ θ)
    (hΔ : ∀ θ ∈ U, |Δ θ| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S, infDist θ S ≤ C * Real.sqrt (FH θ - FH θbar))
    {θstar θbar : Θ} (hstarU : θstar ∈ U) (hbarS : θbar ∈ S) (hbarU : θbar ∈ U)
    (hopt : F θstar ≤ F θbar) :
    infDist θstar S ≤ C * Real.sqrt (2 * M * ε) := by
  calc infDist θstar S ≤ C * Real.sqrt (FH θstar - FH θbar) := herr θstar hstarU θbar hbarS
    _ ≤ C * Real.sqrt (2 * M * ε) := by
        gcongr
        exact head_excess_le hε hF hopt (hΔ θstar hstarU) (hΔ θbar hbarU)

end Perturbation

section Theorem2

variable {Θ : Type*} [PseudoMetricSpace Θ]

/-- The two-set objective `γ (𝓛_H + R) + (1-γ)(𝓛_T + R)` with `γ = k_H IF / (k_H IF + k_T)`. -/
noncomputable def mixObj (LH LT R : Θ → ℝ) (kH kT IF : ℝ) (θ : Θ) : ℝ :=
  (kH * IF / (kH * IF + kT)) * (LH θ + R θ) + (1 - kH * IF / (kH * IF + kT)) * (LT θ + R θ)

/-- **Theorem 3.5** (explicit form, Eq. (l_diff) + (holder)). Under the Hölder-1/2 error bound
for the Head objective `F_H = 𝓛_H + R` on `U ⊇ S_H` and `|𝓛_T - 𝓛_H| ≤ M` on `U`, every global
minimizer `θ*` of the full objective lying in `U` satisfies
`dist(θ*, S_H) ≤ C √(2 M k_T / (k_H IF + k_T))`. -/
theorem theorem3_5_bound {LH LT R : Θ → ℝ} {kH kT IF M C : ℝ} {S U : Set Θ}
    (hkH : 0 < kH) (hkT : 0 ≤ kT) (hIF : 0 < IF) (hC : 0 ≤ C)
    (hM : ∀ θ ∈ U, |LT θ - LH θ| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S,
      infDist θ S ≤ C * Real.sqrt ((LH θ + R θ) - (LH θbar + R θbar)))
    (hSU : S ⊆ U) {θstar θbar : Θ} (hstarU : θstar ∈ U) (hbarS : θbar ∈ S)
    (hstar : ∀ θ, mixObj LH LT R kH kT IF θstar ≤ mixObj LH LT R kH kT IF θ) :
    infDist θstar S ≤ C * Real.sqrt (2 * M * (kT / (kH * IF + kT))) := by
  have hden : 0 < kH * IF + kT := by positivity
  refine dist_le_of_perturbation (FH := fun θ => LH θ + R θ) (Δ := fun θ => LT θ - LH θ)
    (F := mixObj LH LT R kH kT IF) (by positivity) hC ?_ hM herr hstarU hbarS (hSU hbarS)
    (hstar θbar)
  intro θ
  have := one_sub_gamma_eq kH kT IF hden
  unfold mixObj
  rw [this]
  have h : kH * IF / (kH * IF + kT) = 1 - kT / (kH * IF + kT) := by
    rw [← this]; ring
  rw [h]
  ring

/-- **Theorem 3.5** (asymptotic form, Eq. (eq_theorem_non)): `dist(θ*(IF), S_H) = O(1/√IF)`. -/
theorem theorem3_5 {LH LT R : Θ → ℝ} {kH kT M C : ℝ} {S U : Set Θ}
    (hkH : 0 < kH) (hkT : 0 ≤ kT) (hC : 0 ≤ C)
    (hM : ∀ θ ∈ U, |LT θ - LH θ| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S,
      infDist θ S ≤ C * Real.sqrt ((LH θ + R θ) - (LH θbar + R θbar)))
    (hSU : S ⊆ U) (hS : S.Nonempty) (θstar : ℝ → Θ)
    (hstarU : ∀ᶠ IF in atTop, θstar IF ∈ U)
    (hstar : ∀ IF, 0 < IF → ∀ θ, mixObj LH LT R kH kT IF (θstar IF) ≤ mixObj LH LT R kH kT IF θ) :
    (fun IF => infDist (θstar IF) S) =O[atTop] (fun IF => (Real.sqrt IF)⁻¹) := by
  obtain ⟨θbar, hbar⟩ := hS
  have hM0 : ∀ θ ∈ U, 0 ≤ M := fun θ hθ => (abs_nonneg _).trans (hM θ hθ)
  have hM0' : 0 ≤ M := hM0 θbar (hSU hbar)
  refine IsBigO.of_bound (C * Real.sqrt (2 * M * kT / kH)) ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ), hstarU] with IF hIF hU
  rw [Real.norm_of_nonneg infDist_nonneg, Real.norm_of_nonneg (by positivity)]
  calc infDist (θstar IF) S ≤ C * Real.sqrt (2 * M * (kT / (kH * IF + kT))) :=
        theorem3_5_bound hkH hkT hIF hC hM herr hSU hU hbar (hstar IF hIF)
    _ ≤ C * Real.sqrt (2 * M * ((kT / kH) * IF⁻¹)) := by
        gcongr; exact one_sub_gamma_le kH kT IF hkH hkT hIF
    _ = C * Real.sqrt (2 * M * kT / kH) * (Real.sqrt IF)⁻¹ := by
        rw [mul_assoc C, ← Real.sqrt_inv, ← Real.sqrt_mul (by positivity)]
        congr 2; ring

end Theorem2

section Theorem3

variable {Θ : Type*} [PseudoMetricSpace Θ]

/-- **Theorem 3.7** (multiple Head–Tail partitions, Eqs. (eq:aggregate_bound)–(eq_theorem_2)).
Partition sizes `s 0 ≥ s 1 ≥ ⋯ ≥ s (n-1)` with relative tail mass
`ε = ∑_{i ≥ 1} s i / ∑_i s i`. If `F_D = F_{D¹} + ε Δ`, the contrast `Δ` is bounded by `M` on
`U ⊇ S_{D¹}`, the Hölder error bound holds on `U`, and the full minimizer `θ*` lies in `U`, then
`dist(θ*, S_{D¹}) ≤ C √(2Mε) ≤ C √(2 M (n-1) |D²|/|D¹|) = C √(2M(n-1)) · IF_{D¹,D²}^{-1/2}`. -/
theorem theorem3_7 {F1 FD Δ : Θ → ℝ} {M C : ℝ} {S U : Set Θ} (s : ℕ → ℝ) (n : ℕ)
    (hn : 2 ≤ n) (hnonneg : ∀ i, 0 ≤ s i) (hsorted : ∀ i j, i ≤ j → s j ≤ s i)
    (h0 : 0 < s 0) (h1 : 0 < s 1) (hC : 0 ≤ C)
    (hF : ∀ θ, FD θ = F1 θ +
      ((∑ i ∈ Finset.Ico 1 n, s i) / (∑ i ∈ Finset.range n, s i)) * Δ θ)
    (hΔ : ∀ θ ∈ U, |Δ θ| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S, infDist θ S ≤ C * Real.sqrt (F1 θ - F1 θbar))
    (hSU : S ⊆ U) {θstar θbar : Θ} (hstarU : θstar ∈ U) (hbarS : θbar ∈ S)
    (hstar : ∀ θ, FD θstar ≤ FD θ) :
    infDist θstar S ≤
        C * Real.sqrt (2 * M * ((∑ i ∈ Finset.Ico 1 n, s i) / (∑ i ∈ Finset.range n, s i))) ∧
      infDist θstar S ≤ C * Real.sqrt (2 * M * ((n - 1 : ℝ) * s 1 / s 0)) ∧
      infDist θstar S ≤ C * Real.sqrt (2 * M * (n - 1 : ℝ)) * (Real.sqrt (s 0 / s 1))⁻¹ := by
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hΔ θbar (hSU hbarS))
  have hεnn : 0 ≤ (∑ i ∈ Finset.Ico 1 n, s i) / (∑ i ∈ Finset.range n, s i) :=
    div_nonneg (Finset.sum_nonneg fun i _ => hnonneg i) (Finset.sum_nonneg fun i _ => hnonneg i)
  have hA := dist_le_of_perturbation hεnn hC hF hΔ herr hstarU hbarS (hSU hbarS) (hstar θbar)
  have hB : infDist θstar S ≤ C * Real.sqrt (2 * M * ((n - 1 : ℝ) * s 1 / s 0)) := by
    refine hA.trans (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt
      (mul_le_mul_of_nonneg_left (tail_mass_le s n hn hnonneg hsorted h0) (by positivity))) hC)
  refine ⟨hA, hB, hB.trans_eq ?_⟩
  rw [mul_assoc C, ← Real.sqrt_inv, ← Real.sqrt_mul (by
    have : (1 : ℝ) ≤ n := by exact_mod_cast (by omega : 1 ≤ n)
    nlinarith)]
  congr 2
  field_simp

/-- **Theorem 3.7**, data-level form. The dataset is split into pairwise disjoint nonempty
partitions `P 0, …, P (n-1)` sorted by cardinality; the Head is `D¹ = P 0`, the aggregated Tail is
`D_T = ⋃_{i ≥ 1} P i`, and each objective is the average loss plus a shared regularizer `R`.
The objective decomposition `F_D = F_{D¹} + ε (F_{D_T} - F_{D¹})` is derived (not assumed). -/
theorem theorem3_7_data {ι : Type*} [DecidableEq ι] (ℓ : ι → Θ → ℝ) (R : Θ → ℝ)
    (P : ℕ → Finset ι) (n : ℕ) (hn : 2 ≤ n)
    (hdisj : ∀ i j, i ≠ j → Disjoint (P i) (P j)) (hne : ∀ i, i < n → (P i).Nonempty)
    (hsorted : ∀ i j, i ≤ j → (P j).card ≤ (P i).card) {M C : ℝ} {S U : Set Θ} (hC : 0 ≤ C)
    (hΔ : ∀ θ ∈ U, |(avgLoss ℓ ((Finset.Ico 1 n).biUnion P) θ + R θ) -
      (avgLoss ℓ (P 0) θ + R θ)| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S, infDist θ S ≤
      C * Real.sqrt ((avgLoss ℓ (P 0) θ + R θ) - (avgLoss ℓ (P 0) θbar + R θbar)))
    (hSU : S ⊆ U) {θstar θbar : Θ} (hstarU : θstar ∈ U) (hbarS : θbar ∈ S)
    (hstar : ∀ θ, avgLoss ℓ (P 0 ∪ (Finset.Ico 1 n).biUnion P) θstar + R θstar ≤
      avgLoss ℓ (P 0 ∪ (Finset.Ico 1 n).biUnion P) θ + R θ) :
    infDist θstar S ≤
      C * Real.sqrt (2 * M * (n - 1 : ℝ)) * (Real.sqrt (((P 0).card : ℝ) / (P 1).card))⁻¹ := by
  set DT := (Finset.Ico 1 n).biUnion P with hDT
  have hdisjHT : Disjoint (P 0) DT := by
    rw [hDT, Finset.disjoint_biUnion_right]
    intro i hi
    exact hdisj 0 i (by have := (Finset.mem_Ico.1 hi).1; omega)
  have hTne : DT.Nonempty :=
    ⟨_, Finset.mem_biUnion.2 ⟨1, Finset.mem_Ico.2 ⟨le_rfl, by omega⟩,
      (hne 1 (by omega)).choose_spec⟩⟩
  have hcardT : (DT.card : ℝ) = ∑ i ∈ Finset.Ico 1 n, ((P i).card : ℝ) := by
    rw [hDT, Finset.card_biUnion (fun i _ j _ hij => hdisj i j hij)]
    push_cast; rfl
  have hcardU : ((P 0 ∪ DT).card : ℝ) = ∑ i ∈ Finset.range n, ((P i).card : ℝ) := by
    rw [Finset.card_union_of_disjoint hdisjHT]
    push_cast
    rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (by omega : 0 < n), hcardT]
  have hF : ∀ θ, avgLoss ℓ (P 0 ∪ DT) θ + R θ = (avgLoss ℓ (P 0) θ + R θ) +
      ((∑ i ∈ Finset.Ico 1 n, ((P i).card : ℝ)) / (∑ i ∈ Finset.range n, ((P i).card : ℝ))) *
        ((avgLoss ℓ DT θ + R θ) - (avgLoss ℓ (P 0) θ + R θ)) := by
    intro θ
    rw [objective_union ℓ R hdisjHT (hne 0 (by omega)) hTne θ, hcardT, hcardU]
  exact (theorem3_7 (fun i => ((P i).card : ℝ)) n hn (fun i => by positivity)
    (fun i j hij => by exact_mod_cast hsorted i j hij)
    (by exact_mod_cast (hne 0 (by omega)).card_pos)
    (by exact_mod_cast (hne 1 (by omega)).card_pos) hC hF hΔ herr hSU hstarU hbarS
    hstar).2.2

/-- When the Head objective has a unique minimizer, the set distance is the Euclidean distance:
`dist(θ*, {θ*_{D¹}}) = ‖θ* - θ*_{D¹}‖`. -/
theorem infDist_singleton_eq_norm {E : Type*} [NormedAddCommGroup E] (θstar θ1 : E) :
    infDist θstar {θ1} = ‖θstar - θ1‖ := by
  rw [infDist_singleton, dist_eq_norm]

end Theorem3

end CLTR
