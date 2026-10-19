import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np

def analyze_trajectory(topology_path, trajectory_path):
    """Calcula el RMSD de la cadena principal y el radio de giro (Rg) del tetrámero."""
    u = mda.Universe(topology_path, trajectory_path)
    backbone = u.select_atoms("backbone")
    protein = u.select_atoms("protein")
    
    # Análisis sobre la última mitad de la trayectoria de producción
    half_frame = len(u.trajectory) // 2
    
    R = rms.RMSD(backbone, backbone, select="backbone", ref_frame=0).run(start=half_frame)
    mean_rmsd = np.mean(R.results.rmsd[:, 2]) / 10.0  # Conversión de Å a nm
    
    rgs = [protein.radius_of_gyration() / 10.0 for ts in u.trajectory[half_frame:]]
    mean_rg = np.mean(rgs)
    
    print(f"Backbone RMSD: {mean_rmsd:.2f} nm")
    print(f"Tetramer Rg: {mean_rg:.2f} nm")

if __name__ == "__main__":
    pass
