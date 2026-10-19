import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply

# =====================================================================
# 1. PARÁMETROS TERMODINÁMICOS ARN/ADN (Sugimoto et al., 1995; Watkins 2011)
# =====================================================================
SUGIMOTO_DG = {
    'AA': -1.0, 'AU': -0.9, 'AG': -1.8, 'AC': -2.1,
    'UA': -0.6, 'UU': -0.2, 'UG': -1.6, 'UC': -1.7,
    'GA': -1.5, 'GU': -1.3, 'GG': -2.9, 'GC': -2.7,
    'CA': -0.9, 'CU': -0.9, 'CG': -1.7, 'CC': -2.1
}
DG_INIT = +1.9
RT = 0.616  # kcal/mol a 310 K

# Penalizaciones internas dependientes de posición y tipo de par (rX:dY)
WOBBLE_PENALTY = {('U', 'G'): +1.2, ('G', 'T'): +1.3}
MISMATCH_PENALTY = {
    ('A', 'A'): +2.9, ('A', 'C'): +2.7, ('A', 'G'): +2.5,
    ('U', 'C'): +2.8, ('U', 'T'): +2.4,
    ('G', 'A'): +2.4, ('G', 'G'): +2.7,
    ('C', 'A'): +2.8, ('C', 'C'): +3.1, ('C', 'T'): +2.6
}

WT_SPACER = list("UUCUCCACAGGAGUCAGGUG")
WT_WINDOW = list("CUCCAC") # Posiciones 3 a 8
ALPHABET = ['A', 'U', 'G', 'C']
WC_COMP = {'A': 'T', 'U': 'A', 'G': 'C', 'C': 'G'}

HBB_TARGET_DNA = [WC_COMP[b] for b in WT_SPACER]
HBD_TARGET_DNA = HBB_TARGET_DNA.copy()
HBD_TARGET_DNA[6] = 'A'   # Mismatch intrínseco 1 en HBD (pos 7)
HBD_TARGET_DNA[17] = 'G'  # Mismatch intrínseco 2 en HBD (pos 18)

# Hiperparámetros físicos calibrados
DG_TARGET = -25.2   # Umbral de atenuación óptima para 1 mismatch (kcal/mol)
DG_CRIT   = -21.8   # Umbral crítico de apertura de R-loop para Cas9-ABE8e
LAMBDA_OT = 2.5     # Peso de discriminación epistática frente a HBD
LAMBDA_FOLD = 1.2   # Peso de penalización de estructura secundaria/homopolímeros
LAMBDA_CAT = 6.0    # Peso de preservación del sitio catalítico CAC

def calc_hybrid_dg(spacer_seq, dna_target, is_off_target=False):
    """Calcula dG°37 (kcal/mol) incluyendo penalización cooperativa por múltiples mismatches."""
    dg = DG_INIT
    for i in range(19):
        dg += SUGIMOTO_DG[spacer_seq[i] + spacer_seq[i+1]]
    
    n_mm = 0
    for i in range(20):
        r_b, d_b = spacer_seq[i], dna_target[i]
        if WC_COMP[r_b] != d_b:
            n_mm += 1
            if (r_b, d_b) in WOBBLE_PENALTY:
                dg += WOBBLE_PENALTY[(r_b, d_b)]
            else:
                dg += MISMATCH_PENALTY.get((r_b, d_b), +2.7)
                
    if is_off_target and n_mm >= 3:
        dg += 0.85 * (n_mm - 2)**1.5
    return round(dg, 2)

def rloop_occupancy(dg):
    """Probabilidad de formación de R-loop activo según distribución de Fermi-Boltzmann."""
    return 1.0 / (1.0 + np.exp((dg - DG_CRIT) / RT))

def evaluate_qubo_energy(active_seq):
    spacer = WT_SPACER.copy()
    spacer[2:8] = list(active_seq)
    
    dg_hbb = calc_hybrid_dg(spacer, HBB_TARGET_DNA, is_off_target=False)
    dg_hbd = calc_hybrid_dg(spacer, HBD_TARGET_DNA, is_off_target=True)
    
    occ_hbb = rloop_occupancy(dg_hbb)
    occ_hbd = rloop_occupancy(dg_hbd)
    spec_ratio = occ_hbb / max(occ_hbd, 1e-6)
    
    h_rloop = (dg_hbb - DG_TARGET)**2
    h_ot = (100.0 * occ_hbd)**2
    
    sub_str = "".join(active_seq)
    h_fold = 2.0 if ("UUU" in sub_str or "CCC" in sub_str or "GGG" in sub_str or "AAA" in sub_str) else 0.0
    
    n_mut_distal = sum(1 for k in range(3) if active_seq[k] != WT_WINDOW[k])
    n_mut_cat    = sum(1 for k in range(3, 6) if active_seq[k] != WT_WINDOW[k])
    
    h_cat = 8.0 * n_mut_cat + 2.2 * (n_mut_distal - 1)**2
    if n_mut_distal == 1 and active_seq[2] == WT_WINDOW[2]:
        h_cat += 0.45
        
    total_E = h_rloop + LAMBDA_OT * h_ot + LAMBDA_FOLD * h_fold + LAMBDA_CAT * h_cat
    return total_E, dg_hbb, dg_hbd, spec_ratio

# =====================================================================
# 2. ENUMERACIÓN EXHAUSTIVA (4096 ESTADOS)
# =====================================================================
all_states, energies = [], []
meta_info = {}

for idx in range(4096):
    rem = idx
    seq = [ALPHABET[(rem // (4**k)) % 4] for k in range(6)]
    seq_str = "".join(seq)
    E, dg_b, dg_d, spec = evaluate_qubo_energy(seq)
    all_states.append(seq_str)
    energies.append(E)
    meta_info[seq_str] = (E, dg_b, dg_d, spec)

energies = np.array(energies)
E_min, E_max = energies.min(), energies.max()
sorted_indices = np.argsort(energies)

print("=== [RESULTADOS ENUMERACIÓN EXHAUSTIVA] ===")
print(f"E_min = {E_min:.4f} | E_max = {E_max:.4f}")
