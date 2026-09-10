"""Generate the small astronomy markdown dataset used by the tutorial."""

import os

docs: dict[str, str] = {
    "black_holes.md": """# Black Holes
Black holes are regions of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from them. 
The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to the extent that it creates a black hole.
There are four main types of black holes: stellar-mass, intermediate-mass, supermassive, and primordial.
Supermassive black holes are found at the center of almost every large galaxy, including our own Milky Way (Sagittarius A*).
The event horizon is the boundary around a black hole beyond which no information can return.""",
    "nebulae.md": """# Nebulae
A nebula is an interstellar cloud of dust, hydrogen, helium and other ionized gases. 
Nebulae are often "stellar nurseries," regions where new stars are born from the collapse of dense gas clouds.
Types of nebulae include:
1. Emission nebulae: Glow because they are ionized by nearby hot stars.
2. Reflection nebulae: Reflect light from nearby stars.
3. Dark nebulae: Block light from objects behind them.
4. Planetary nebulae: Created when a star like the Sun expires and sheds its outer layers.""",
    "galaxy_clusters.md": """# Galaxy Clusters
Galaxy clusters are the largest gravitationally bound structures in the universe. 
They consist of hundreds to thousands of galaxies held together by dark matter.
The majority of the mass in a cluster is composed of dark matter, while most of the visible mass (excluding stars) is hot, X-ray emitting gas called the intracluster medium.
Clusters are found at the intersections of cosmic filaments in the large-scale structure of the universe.""",
    "dark_energy.md": """# Dark Energy
Dark energy is a theoretical form of energy that permeates all of space and tends to accelerate the expansion of the universe. 
It makes up approximately 68% of the total energy density of the universe.
While dark matter acts as an attractive force, dark energy acts as a repulsive one on cosmological scales.
The discovery of accelerating expansion was made via observations of Type Ia supernovae.""",
    "exoplanets.md": """# Exoplanets
An exoplanet is a planet that orbits a star outside our solar system. 
Methods for detection include:
- Transit Method: Observing the dip in brightness as a planet passes in front of its star.
- Radial Velocity: Detecting the wobble of a star due to the gravitational pull of an orbiting planet.
- Direct Imaging: Capturing actual images of planets by blocking out the light from the host star.
The "habitable zone" is the region around a star where liquid water could potentially exist on a planet's surface.""",
}

os.makedirs("dataset/astronomy_docs", exist_ok=True)

for filename, content in docs.items():
    with open(f"dataset/astronomy_docs/{filename}", "w", encoding="utf-8") as f:
        f.write(content)

print(f"Generated {len(docs)} astronomy documents in dataset/astronomy_docs")
