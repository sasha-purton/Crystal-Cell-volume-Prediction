from mp_api.client import MPRester
import joblib, os
from tqdm import tqdm
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure, Element

calc = XRDCalculator(wavelength="CuKa")

def get_xrd_pattern(structure: Structure, two_theta_range=(10, 80), dim=1024):
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)
    x = torch.linspace(two_theta_range[0], two_theta_range[1], dim)
    # Gaussian broaden peaks
    spectrum = torch.zeros_like(x)
    for peak_x, intensity in zip(pattern.x, pattern.y):
        idx = torch.argmin(abs(x - peak_x))
        spectrum[idx] = intensity
    return spectrum / spectrum.max()

def composition_vector(structure: Structure):
    """Encodes composition as normalized atomic fractions in periodic-table order."""
    comp = structure.composition
    vec = torch.zeros(100)  # up to element Z=100
    for el, amt in comp.get_el_amt_dict().items():
        el = Element(el)
        vec[el.Z - 1] = amt
    return vec / torch.sum(vec)

mpr = MPRester("<api key>")
SAVE_DIR = "directory"
os.makedirs(SAVE_DIR, exist_ok=True)

results = mpr.materials.summary.search(
    crystal_system=["cubic"],
    fields=["structure", "volume"]
)

data = []

for doc in tqdm(results):
    s = doc.structure
    try:
        xrd = get_xrd_pattern(s)
        comp_vec = composition_vector(s)
        volume = doc.volume
        data.append({
            "volume": volume,
            "xrd": xrd,
            "composition": comp_vec
        })
    except Exception as e:
        print(f"Skipping {doc.material_id}: {e}")

joblib.dump(data, "cubic_xrd_dataset.pkl", compress=3)