/-
# Convex case: Lemma C.1 and Theorem 3.3

Lemma C.1 is proved from *quadratic growth* of `f` and `g` at their minimizers:

  `f x_f + λ_f/2 ‖y - x_f‖² ≤ f y`   and   `g x_g + λ_g/2 ‖y - x_g‖² ≤ g y`   for all `y`.

Quadratic growth holds for every `λ`-strongly convex
function at its minimizer (`quadratic_growth_of_strongConvexOn`), in particular for a convex loss
plus the `L²` regularizer `μ/2 ‖θ‖²` with `λ = μ` (Assumption 3.2).
-/
import Mathlib
import Proofs.Data

open Set Filter Asymptotics Topology

namespace CLTR

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A `m`-strongly convex function grows at least quadratically away from a global minimizer. -/
theorem quadratic_growth_of_strongConvexOn {f : E → ℝ} {m : ℝ} (hm : 0 ≤ m)
    (hf : StrongConvexOn univ m f) {x : E} (hmin : ∀ y, f x ≤ f y) (y : E) :
    f x + m / 2 * ‖y - x‖ ^ 2 ≤ f y := by
  set K := m / 2 * ‖y - x‖ ^ 2 with hK
  have hK0 : 0 ≤ K := by positivity
  by_contra hcon
  rw [not_le] at hcon
  set g := f x + K - f y with hg
  have hg0 : 0 < g := by linarith
  set t := g / (2 * (K + g)) with ht
  have hKg : 0 < K + g := by linarith
  have ht0 : 0 < t := by positivity
  have ht1 : t < 1 := by
    rw [ht, div_lt_one (by positivity)]
    linarith
  have htdef : t * (2 * (K + g)) = g := by rw [ht]; field_simp
  have key := hf.2 (mem_univ x) (mem_univ y) (by linarith : (0 : ℝ) ≤ 1 - t) ht0.le
    (by ring : 1 - t + t = 1)
  simp only [smul_eq_mul] at key
  have hxy : ‖x - y‖ = ‖y - x‖ := norm_sub_rev x y
  rw [hxy] at key
  have h1 := hmin ((1 - t) • x + t • y)
  -- `t f x ≤ t f y - (1-t) t K`, hence `g ≤ t K`
  have h2 : t * g ≤ t * (t * K) := by
    have : f x ≤ (1 - t) * f x + t * f y - (1 - t) * t * K := by
      rw [hK]; linarith
    nlinarith
  have h3 : g ≤ t * K := le_of_mul_le_mul_left h2 ht0
  nlinarith

/-- **Lemma C.1** (rigorous form). If `|f - g| ≤ δ` everywhere and `f`, `g` grow quadratically
with constants `λ_f`, `λ_g` around their minimizers `x_f`, `x_g`, then
`(λ_f + λ_g) ‖x_f - x_g‖² ≤ 4 δ`. -/
theorem lemmaC1 {f g : E → ℝ} {xf xg : E} {lf lg δ : ℝ}
    (hf : ∀ y, f xf + lf / 2 * ‖y - xf‖ ^ 2 ≤ f y)
    (hg : ∀ y, g xg + lg / 2 * ‖y - xg‖ ^ 2 ≤ g y)
    (hδ : ∀ x, |f x - g x| ≤ δ) :
    (lf + lg) * ‖xf - xg‖ ^ 2 ≤ 4 * δ := by
  have h1 := hf xg
  have h2 := hg xf
  have h3 := (abs_le.1 (hδ xg)).2
  have h4 := (abs_le.1 (hδ xf)).1
  rw [norm_sub_rev xg xf] at h1
  nlinarith

/-- **Lemma C.1** in the paper's form: `‖x_f - x_g‖² ≤ 4δ / (λ_f + λ_g)`. -/
theorem lemmaC1' {f g : E → ℝ} {xf xg : E} {lf lg δ : ℝ} (hpos : 0 < lf + lg)
    (hf : ∀ y, f xf + lf / 2 * ‖y - xf‖ ^ 2 ≤ f y)
    (hg : ∀ y, g xg + lg / 2 * ‖y - xg‖ ^ 2 ≤ g y)
    (hδ : ∀ x, |f x - g x| ≤ δ) :
    ‖xf - xg‖ ^ 2 ≤ 4 * δ / (lf + lg) := by
  rw [le_div_iff₀ hpos, mul_comm]
  exact lemmaC1 hf hg hδ

/-- A convex function plus the `L²` regularizer `μ/2‖θ‖²` is `μ`-strongly convex. -/
theorem strongConvexOn_add_sq_norm {L : E → ℝ} (hL : ConvexOn ℝ univ L) (μ : ℝ) :
    StrongConvexOn univ μ (fun θ => L θ + μ / 2 * ‖θ‖ ^ 2) := by
  rw [strongConvexOn_iff_convex]
  simpa using hL

/-- Mixing weight of the Head loss, Eq. (gamma_if): `γ = k_H IF / (k_H IF + k_T)`. -/
noncomputable def gammaIF (kH kT IF : ℝ) : ℝ := kH * IF / (kH * IF + kT)

/-- The regularized full-data objective `γ 𝓛_H + (1-γ) 𝓛_T + μ/2 ‖θ‖²`
(Eqs. (loss_t), (loss_4)). -/
noncomputable def fullObj (LH LT : E → ℝ) (μ γ : ℝ) (θ : E) : ℝ :=
  γ * LH θ + (1 - γ) * LT θ + μ / 2 * ‖θ‖ ^ 2

/-- The regularized Head-only objective `𝓛_H + μ/2 ‖θ‖²`. -/
noncomputable def headObj (LH : E → ℝ) (μ : ℝ) (θ : E) : ℝ :=
  LH θ + μ / 2 * ‖θ‖ ^ 2

/-- Eq. (loss_5_0_1): `|F - F_H| = (1-γ) |𝓛_T - 𝓛_H| ≤ M k_T / (k_H IF + k_T)`. -/
theorem fullObj_sub_headObj_le (LH LT : E → ℝ) (μ : ℝ) {kH kT IF M : ℝ}
    (hkH : 0 < kH) (hkT : 0 ≤ kT) (hIF : 0 < IF)
    (hM : ∀ θ, |LT θ - LH θ| ≤ M) (θ : E) :
    |fullObj LH LT μ (gammaIF kH kT IF) θ - headObj LH μ θ| ≤ M * (kT / (kH * IF + kT)) := by
  have hden : 0 < kH * IF + kT := by positivity
  have h1γ : 1 - gammaIF kH kT IF = kT / (kH * IF + kT) := one_sub_gamma_eq kH kT IF hden
  have hexp : fullObj LH LT μ (gammaIF kH kT IF) θ - headObj LH μ θ =
      (1 - gammaIF kH kT IF) * (LT θ - LH θ) := by
    unfold fullObj headObj; ring
  rw [hexp, h1γ, abs_mul, abs_of_nonneg (by positivity), mul_comm]
  exact mul_le_mul_of_nonneg_right (hM θ) (by positivity)

/-- **Theorem 3.3** (explicit form, Eq. (way_to_calc)). Under Assumption 3.2, with convex Head and
Tail losses, `L²` weight `μ > 0`, and bounded loss contrast `|𝓛_T - 𝓛_H| ≤ M`, the minimizer
`θ*` of the full objective and the minimizer `θ*_H` of the Head objective satisfy
`‖θ* - θ*_H‖² ≤ 2 δ / μ` with `δ = M k_T / (k_H IF + k_T)`. -/
theorem theorem3_3_sq {LH LT : E → ℝ} (hLH : ConvexOn ℝ univ LH) (hLT : ConvexOn ℝ univ LT)
    {μ kH kT IF M : ℝ} (hμ : 0 < μ) (hkH : 0 < kH) (hkT : 0 ≤ kT) (hIF : 0 < IF)
    (hM : ∀ θ, |LT θ - LH θ| ≤ M) {θstar θH : E}
    (hstar : ∀ θ, fullObj LH LT μ (gammaIF kH kT IF) θstar ≤
      fullObj LH LT μ (gammaIF kH kT IF) θ)
    (hH : ∀ θ, headObj LH μ θH ≤ headObj LH μ θ) :
    ‖θstar - θH‖ ^ 2 ≤ 2 * (M * (kT / (kH * IF + kT))) / μ := by
  have hden : 0 < kH * IF + kT := by positivity
  have hγ0 : 0 ≤ gammaIF kH kT IF := by unfold gammaIF; positivity
  have h1γ0 : 0 ≤ 1 - gammaIF kH kT IF := by
    rw [gammaIF, one_sub_gamma_eq kH kT IF hden]; positivity
  -- both objectives are `μ`-strongly convex
  have hmix : ConvexOn ℝ univ (fun θ => gammaIF kH kT IF * LH θ +
      (1 - gammaIF kH kT IF) * LT θ) := by
    have := (hLH.smul hγ0).add (hLT.smul h1γ0)
    simpa [Pi.add_def, smul_eq_mul] using this
  have hsF : StrongConvexOn univ μ (fullObj LH LT μ (gammaIF kH kT IF)) := by
    exact strongConvexOn_add_sq_norm hmix μ
  have hsH : StrongConvexOn univ μ (headObj LH μ) := by
    exact strongConvexOn_add_sq_norm hLH μ
  have gF := quadratic_growth_of_strongConvexOn hμ.le hsF hstar
  have gH := quadratic_growth_of_strongConvexOn hμ.le hsH hH
  have := lemmaC1' (by linarith) gF gH (fullObj_sub_headObj_le LH LT μ hkH hkT hIF hM)
  calc ‖θstar - θH‖ ^ 2 ≤ 4 * (M * (kT / (kH * IF + kT))) / (μ + μ) := this
    _ = 2 * (M * (kT / (kH * IF + kT))) / μ := by field_simp; ring

/-- **Theorem 3.3** (explicit rate): `‖θ* - θ*_H‖ ≤ √(2 M k_T / (μ k_H)) · IF^{-1/2}`. -/
theorem theorem3_3_bound {LH LT : E → ℝ} (hLH : ConvexOn ℝ univ LH) (hLT : ConvexOn ℝ univ LT)
    {μ kH kT IF M : ℝ} (hμ : 0 < μ) (hkH : 0 < kH) (hkT : 0 ≤ kT) (hIF : 0 < IF)
    (hM : ∀ θ, |LT θ - LH θ| ≤ M) {θstar θH : E}
    (hstar : ∀ θ, fullObj LH LT μ (gammaIF kH kT IF) θstar ≤
      fullObj LH LT μ (gammaIF kH kT IF) θ)
    (hH : ∀ θ, headObj LH μ θH ≤ headObj LH μ θ) :
    ‖θstar - θH‖ ≤ Real.sqrt (2 * M * kT / (μ * kH)) * (Real.sqrt IF)⁻¹ := by
  have hsq := theorem3_3_sq hLH hLT hμ hkH hkT hIF hM hstar hH
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM 0)
  have hle : 2 * (M * (kT / (kH * IF + kT))) / μ ≤ 2 * M * kT / (μ * kH) * IF⁻¹ := by
    have := one_sub_gamma_le kH kT IF hkH hkT hIF
    calc 2 * (M * (kT / (kH * IF + kT))) / μ ≤ 2 * (M * ((kT / kH) * IF⁻¹)) / μ := by gcongr
      _ = 2 * M * kT / (μ * kH) * IF⁻¹ := by field_simp
  rw [← Real.sqrt_inv, ← Real.sqrt_mul (by positivity)]
  exact Real.le_sqrt_of_sq_le (hsq.trans hle)

/-- **Theorem 3.3** (asymptotic form, Eq. (eq13)): for any choice `θ*(IF)` of full-data minimizers
(one for each imbalance factor), `‖θ*(IF) - θ*_H‖ = O(1/√IF)` as `IF → ∞`. -/
theorem theorem3_3 {LH LT : E → ℝ} (hLH : ConvexOn ℝ univ LH) (hLT : ConvexOn ℝ univ LT)
    {μ kH kT M : ℝ} (hμ : 0 < μ) (hkH : 0 < kH) (hkT : 0 ≤ kT)
    (hM : ∀ θ, |LT θ - LH θ| ≤ M) {θH : E} (hH : ∀ θ, headObj LH μ θH ≤ headObj LH μ θ)
    (θstar : ℝ → E)
    (hstar : ∀ IF, 0 < IF → ∀ θ, fullObj LH LT μ (gammaIF kH kT IF) (θstar IF) ≤
      fullObj LH LT μ (gammaIF kH kT IF) θ) :
    (fun IF => ‖θstar IF - θH‖) =O[atTop] (fun IF => (Real.sqrt IF)⁻¹) := by
  refine IsBigO.of_bound (Real.sqrt (2 * M * kT / (μ * kH))) ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with IF hIF
  rw [Real.norm_of_nonneg (norm_nonneg _), Real.norm_of_nonneg (by positivity)]
  exact theorem3_3_bound hLH hLT hμ hkH hkT hIF hM (hstar IF hIF) hH

end CLTR
