"""
RS-Specific Prompt Axes Definition for Systematic Prompt Optimization.

Adapted from Botanical Grammar (Lundqvist et al., CVPRW 2026):
- 5 axes + negation x 20 DIOR classes
- Each axis has a baseline + 2-3 perturbation levels
- Category-specific values designed from remote sensing overhead perspective

Axes:
  1. Granularity (3 levels): coarse / standard(baseline) / fine
  2. Spatial Context (3 levels): none(baseline) / location / context
  3. Geometric Descriptor (3 levels): none(baseline) / shape / geometry
  4. Scale Descriptor (3 levels): none(baseline) / size / comparative
  5. Grammar/Viewpoint (4 levels): bare(baseline) / photo_of / aerial / satellite
  6. Negation (2 levels): none(baseline) / negation (Phase 2 only)

Prompt template: [grammar_prefix] [scale_adj] [noun_spatial_geo_core] [negation?]
"""

# ============================================================
# Axis 1: Semantic Granularity (from existing dior-granularity-prompts.md)
# ============================================================
GRANULARITY = {
    "airplane": {"coarse": "aircraft", "standard": "airplane",
                 "fine": "fixed-wing passenger airplane on tarmac"},
    "airport": {"coarse": "aviation facility", "standard": "airport",
                "fine": "civil airport with runways and terminal buildings"},
    "baseball_field": {"coarse": "sports field", "standard": "baseball field",
                       "fine": "baseball diamond with infield dirt and outfield grass"},
    "basketball_court": {"coarse": "sports court", "standard": "basketball court",
                         "fine": "rectangular basketball court with white court markings"},
    "bridge": {"coarse": "bridge or viaduct", "standard": "bridge",
               "fine": "highway bridge spanning across water"},
    "chimney": {"coarse": "tall industrial structure", "standard": "chimney",
                "fine": "industrial smokestack chimney with elongated shadow"},
    "dam": {"coarse": "water barrier structure", "standard": "dam",
            "fine": "concrete dam wall crossing a river"},
    "expressway_service_area": {"coarse": "roadside facility complex", "standard": "expressway service area",
                                "fine": "highway rest area with parking lot and multiple buildings"},
    "expressway_toll_station": {"coarse": "roadside toll facility", "standard": "expressway toll station",
                                "fine": "highway toll plaza with multiple parallel toll booths and canopies"},
    "golf_course": {"coarse": "recreational green space", "standard": "golf course",
                    "fine": "golf course with fairways and sand bunkers"},
    "ground_track_field": {"coarse": "sports field", "standard": "ground track field",
                           "fine": "athletic field with oval running track"},
    "harbor": {"coarse": "waterfront facility", "standard": "harbor",
               "fine": "maritime harbor with docks and berthed ships"},
    "overpass": {"coarse": "elevated roadway intersection", "standard": "overpass",
                 "fine": "cloverleaf or diamond highway interchange"},
    "ship": {"coarse": "vessel", "standard": "ship",
             "fine": "large cargo vessel or container ship on water"},
    "stadium": {"coarse": "large sports venue", "standard": "stadium",
                "fine": "large stadium with oval spectator seating stands"},
    "storage_tank": {"coarse": "circular industrial structure", "standard": "storage tank",
                     "fine": "cylindrical oil storage tank with floating roof"},
    "tennis_court": {"coarse": "sports court", "standard": "tennis court",
                     "fine": "rectangular tennis court with white boundary lines and net"},
    "train_station": {"coarse": "transportation hub", "standard": "train station",
                      "fine": "railway station with multiple tracks and platforms"},
    "vehicle": {"coarse": "motor vehicle", "standard": "vehicle",
                "fine": "small passenger car on road or parking lot"},
    "windmill": {"coarse": "energy infrastructure", "standard": "windmill",
                 "fine": "three-blade wind turbine with white tower"},
}

# ============================================================
# Axis 2: Spatial Context
# ============================================================
SPATIAL = {
    "airplane": {"none": "", "location": "on tarmac",
                 "context": "parked at airport apron near terminal"},
    "airport": {"none": "", "location": "in open area",
                "context": "with intersecting runways surrounded by grassland"},
    "baseball_field": {"none": "", "location": "in urban area",
                       "context": "in urban park or school grounds surrounded by buildings"},
    "basketball_court": {"none": "", "location": "on paved ground",
                         "context": "in residential area or school playground"},
    "bridge": {"none": "", "location": "over water",
               "context": "spanning across river or strait connecting two shores"},
    "chimney": {"none": "", "location": "at industrial plant",
                "context": "standing at industrial facility or power plant"},
    "dam": {"none": "", "location": "across river",
            "context": "spanning across river valley between two hillsides"},
    "expressway_service_area": {"none": "", "location": "along highway",
                                "context": "adjacent to highway with entry and exit ramps"},
    "expressway_toll_station": {"none": "", "location": "on highway",
                                "context": "spanning across highway lanes with widened road section"},
    "golf_course": {"none": "", "location": "in suburban area",
                    "context": "in suburban green belt with surrounding vegetation"},
    "ground_track_field": {"none": "", "location": "in school or park",
                           "context": "in school campus or public sports complex"},
    "harbor": {"none": "", "location": "at coastline",
               "context": "along coastline or riverbank with piers extending into water"},
    "overpass": {"none": "", "location": "at highway junction",
                 "context": "at intersection of multiple highways forming interchange"},
    "ship": {"none": "", "location": "on water",
             "context": "sailing in open water or berthed at dock"},
    "stadium": {"none": "", "location": "in urban center",
                "context": "in urban area with large surrounding parking lots"},
    "storage_tank": {"none": "", "location": "in industrial zone",
                     "context": "grouped in oil refinery or chemical plant facility"},
    "tennis_court": {"none": "", "location": "on paved ground",
                     "context": "in residential area or sports club compound"},
    "train_station": {"none": "", "location": "in urban area",
                      "context": "in city center with converging rail tracks and large platform area"},
    "vehicle": {"none": "", "location": "on road",
                "context": "driving on road or parked in parking lot"},
    "windmill": {"none": "", "location": "in open field",
                 "context": "in open farmland or coastal area, often in regular grid arrays"},
}

# ============================================================
# Axis 3: Geometric Descriptor (top-down visible shape)
# ============================================================
GEOMETRIC = {
    "airplane": {"none": "", "shape": "elongated with swept-back wings",
                 "geometry": "elongated fuselage with swept-back wings and T-tail, cruciform shape"},
    "airport": {"none": "", "shape": "large complex with straight intersecting lines",
                "geometry": "large area with intersecting straight runways forming cross or V patterns"},
    "baseball_field": {"none": "", "shape": "diamond-shaped field",
                       "geometry": "diamond-shaped infield with fan-shaped outfield and visible base paths"},
    "basketball_court": {"none": "", "shape": "rectangular paved court",
                         "geometry": "rectangular paved court with center circle and key markings visible"},
    "bridge": {"none": "", "shape": "linear structure over water",
               "geometry": "straight linear deck across water with parallel road lanes"},
    "chimney": {"none": "", "shape": "tall slender cylinder",
                "geometry": "tall circular cross-section tower with distinctive elongated shadow"},
    "dam": {"none": "", "shape": "straight wall across river",
            "geometry": "straight concrete wall cutting perpendicular across river channel"},
    "expressway_service_area": {"none": "", "shape": "roadside complex with buildings",
                                "geometry": "cluster of rectangular buildings alongside highway with large paved parking area"},
    "expressway_toll_station": {"none": "", "shape": "widened road section with booths",
                                "geometry": "road widening with multiple parallel lanes and toll booth canopies"},
    "golf_course": {"none": "", "shape": "irregular green area with smooth fairways",
                    "geometry": "irregular shaped green areas with smooth fairways and scattered circular sand bunkers"},
    "ground_track_field": {"none": "", "shape": "oval with dark track ring",
                           "geometry": "oval running track encircling rectangular grass field, track appears as dark ring"},
    "harbor": {"none": "", "shape": "irregular water edge with rectangular piers",
               "geometry": "irregular coastline with rectangular piers extending into water, ships alongside docks"},
    "overpass": {"none": "", "shape": "curved multi-level road junction",
                 "geometry": "cloverleaf or diamond shaped multi-level road intersection with looping ramps"},
    "ship": {"none": "", "shape": "elongated hull on water",
             "geometry": "elongated rectangular hull with pointed bow, visible wake in water"},
    "stadium": {"none": "", "shape": "large oval or circular structure",
                "geometry": "large oval structure with central opening, concentric ring pattern from above"},
    "storage_tank": {"none": "", "shape": "circular structure with central dot",
                     "geometry": "cylindrical with circular top, floating roof visible as concentric circles"},
    "tennis_court": {"none": "", "shape": "rectangular court with net line",
                     "geometry": "rectangular court with center net line and service boxes visible as white markings"},
    "train_station": {"none": "", "shape": "elongated with parallel rail tracks",
                      "geometry": "cluster of buildings alongside multiple parallel rail tracks converging at station"},
    "vehicle": {"none": "", "shape": "small rectangular shape",
                "geometry": "small rectangular shape from above, much smaller than surrounding structures"},
    "windmill": {"none": "", "shape": "three-blade pinwheel structure",
                 "geometry": "three slender blades radiating from central hub, white tower base with long shadow"},
}

# ============================================================
# Axis 4: Scale Descriptor (modifiers only, avoid noun repetition)
# ============================================================
SCALE = {
    "airplane": {"none": "", "size": "medium-sized",
                 "comparative": "medium-sized, dwarfed by surrounding runways"},
    "airport": {"none": "", "size": "large-scale",
                "comparative": "large-scale, dominating the surrounding landscape"},
    "baseball_field": {"none": "", "size": "moderate-sized",
                       "comparative": "moderate-sized, comparable to nearby buildings"},
    "basketball_court": {"none": "", "size": "small",
                         "comparative": "small, much smaller than surrounding buildings"},
    "bridge": {"none": "", "size": "long",
               "comparative": "long linear, spanning across wide water body"},
    "chimney": {"none": "", "size": "tall slender",
                "comparative": "tall slender, towering above surrounding industrial buildings"},
    "dam": {"none": "", "size": "massive",
            "comparative": "massive, spanning entire river width"},
    "expressway_service_area": {"none": "", "size": "medium-sized",
                                "comparative": "medium-sized, distinct along highway"},
    "expressway_toll_station": {"none": "", "size": "widened",
                                "comparative": "widened, with multiple parallel lanes"},
    "golf_course": {"none": "", "size": "extensive",
                    "comparative": "extensive landscaped, one of the largest green spaces visible"},
    "ground_track_field": {"none": "", "size": "moderate-sized",
                           "comparative": "moderate-sized oval, comparable to a school building"},
    "harbor": {"none": "", "size": "large-scale",
               "comparative": "large-scale waterfront, extending along the coast"},
    "overpass": {"none": "", "size": "large",
                 "comparative": "large interchange, spanning multiple highway levels"},
    "ship": {"none": "", "size": "large",
             "comparative": "large vessel, prominent feature on water surface"},
    "stadium": {"none": "", "size": "very large",
                "comparative": "very large oval, one of the largest individual structures visible"},
    "storage_tank": {"none": "", "size": "medium-sized circular",
                     "comparative": "medium to large circular, typically 20 to 50 meters across"},
    "tennis_court": {"none": "", "size": "small",
                     "comparative": "small rectangular, similar in size to a house"},
    "train_station": {"none": "", "size": "large",
                      "comparative": "large complex, prominent in urban landscape"},
    "vehicle": {"none": "", "size": "tiny",
                "comparative": "tiny, much smaller than surrounding buildings and roads"},
    "windmill": {"none": "", "size": "tall",
                 "comparative": "tall slender, with visible shadow extending from base"},
}

# ============================================================
# Axis 5: Spectral/Color Descriptor (导师建议新增 — RS核心特征)
# none(基线) / color(basic color/appearance) / spectral(detailed spectral appearance)
# ============================================================
SPECTRAL = {
    "airplane": {"none": "", "color": "white or silver",
                 "spectral": "bright white or metallic silver fuselage, high contrast against dark tarmac"},
    "airport": {"none": "", "color": "gray with dark runways",
                "spectral": "gray concrete with dark asphalt runways, surrounded by green or brown vegetation"},
    "baseball_field": {"none": "", "color": "green with brown patch",
                       "spectral": "bright green outfield grass with reddish-brown infield dirt, high contrast"},
    "basketball_court": {"none": "", "color": "dark gray or green",
                         "spectral": "uniform dark gray or dark green paved surface with bright white painted lines"},
    "bridge": {"none": "", "color": "gray or white",
               "spectral": "light gray or white concrete deck, dark shadow cast on water below"},
    "chimney": {"none": "", "color": "gray or reddish",
                "spectral": "gray concrete or reddish brick, with dark elongated shadow on one side"},
    "dam": {"none": "", "color": "light gray concrete",
            "spectral": "bright light gray concrete wall, dark blue or brown water on one side, lighter on the other"},
    "expressway_service_area": {"none": "", "color": "gray with dark roofs",
                                "spectral": "gray asphalt parking area with dark or red roofed buildings"},
    "expressway_toll_station": {"none": "", "color": "gray with white canopies",
                                "spectral": "dark gray asphalt with bright white toll booth canopies"},
    "golf_course": {"none": "", "color": "green with white spots",
                    "spectral": "bright green fairways with small bright white sand bunkers, darker rough areas"},
    "ground_track_field": {"none": "", "color": "reddish oval with green center",
                           "spectral": "reddish-brown oval running track surrounding bright green grass field"},
    "harbor": {"none": "", "color": "gray docks on blue water",
               "spectral": "gray concrete piers extending into dark blue or brown water"},
    "overpass": {"none": "", "color": "gray curved roads",
                 "spectral": "light gray concrete roadways forming curved interchanges, dark shadows underneath"},
    "ship": {"none": "", "color": "dark gray or red",
             "spectral": "dark gray or red hull, bright white deck structures, visible against blue or dark water"},
    "stadium": {"none": "", "color": "gray with dark center",
                "spectral": "light gray or white roof structure with dark oval opening in center"},
    "storage_tank": {"none": "", "color": "white or silver circular",
                     "spectral": "bright white or silver circular tops with dark central dot, high contrast against ground"},
    "tennis_court": {"none": "", "color": "dark green or blue",
                     "spectral": "uniform dark green or blue court surface with bright white boundary lines"},
    "train_station": {"none": "", "color": "gray with silver tracks",
                      "spectral": "gray platform buildings alongside bright silver parallel rail tracks"},
    "vehicle": {"none": "", "color": "various dark or light",
                "spectral": "small dark or light colored rectangle, varies by vehicle type, distinctive against road"},
    "windmill": {"none": "", "color": "bright white",
                 "spectral": "bright white tower and blades, high contrast against green or brown ground, long dark shadow"},
}

# ============================================================
# Axis 6: Texture Descriptor (导师建议新增 — RS表面纹理特征)
# none(基线) / texture(simple texture) / pattern(detailed surface pattern)
# ============================================================
TEXTURE = {
    "airplane": {"none": "", "texture": "smooth metallic surface",
                 "pattern": "smooth uniform fuselage with distinct wing surface, bright specular highlights"},
    "airport": {"none": "", "texture": "mixed smooth and rough",
                "pattern": "smooth runway strips alternating with rougher grassland texture"},
    "baseball_field": {"none": "", "texture": "smooth grass with granular dirt",
                       "pattern": "smooth uniform outfield grass contrasting with granular textured infield dirt"},
    "basketball_court": {"none": "", "texture": "smooth uniform paved",
                         "pattern": "smooth paved court surface with narrow painted line markings"},
    "bridge": {"none": "", "texture": "smooth road surface",
               "pattern": "smooth uniform road deck with parallel lane markings, dark shadow edge"},
    "chimney": {"none": "", "texture": "smooth cylindrical surface",
                "pattern": "smooth cylindrical surface, distinctive elongated shadow creating sharp contrast edge"},
    "dam": {"none": "", "texture": "smooth concrete wall",
            "pattern": "smooth uniform concrete wall surface, sharp straight edge against water"},
    "expressway_service_area": {"none": "", "texture": "mixed paved and built-up",
                                "pattern": "large uniform asphalt parking area adjacent to clustered building roofs"},
    "expressway_toll_station": {"none": "", "texture": "striped road surface",
                                "pattern": "parallel striped road lanes with evenly spaced canopy structures"},
    "golf_course": {"none": "", "texture": "smooth grass with granular spots",
                    "pattern": "smooth fairway grass with small granular bright sand bunker spots, rougher border vegetation"},
    "ground_track_field": {"none": "", "texture": "smooth track with grass center",
                           "pattern": "uniform smooth oval track ring surrounding textured green grass field"},
    "harbor": {"none": "", "texture": "smooth water with structured docks",
               "pattern": "smooth uniform water surface with rectangular structured dock texture"},
    "overpass": {"none": "", "texture": "smooth road with shadow edges",
                 "pattern": "smooth curving road surface with sharp dark shadow edges underneath overpasses"},
    "ship": {"none": "", "texture": "smooth hull with deck structures",
             "pattern": "smooth elongated hull surface with small rectangular deck structure patterns on top"},
    "stadium": {"none": "", "texture": "smooth roof with central opening",
                "pattern": "smooth uniform roof surface with textured central opening revealing seating or field"},
    "storage_tank": {"none": "", "texture": "smooth circular tops",
                     "pattern": "smooth circular tops with small central dot, often in regularly spaced cluster pattern"},
    "tennis_court": {"none": "", "texture": "smooth uniform court surface",
                     "pattern": "smooth uniform court surface with thin white line markings dividing service boxes"},
    "train_station": {"none": "", "texture": "parallel linear rail pattern",
                      "pattern": "multiple parallel thin rail lines converging, alongside smooth platform surfaces"},
    "vehicle": {"none": "", "texture": "small smooth rectangle",
                "pattern": "small smooth rectangular surface, distinct from textured road surface"},
    "windmill": {"none": "", "texture": "smooth tower with blade pattern",
                 "pattern": "smooth white tower base with three distinct blade surfaces radiating from hub"},
}

# ============================================================
# Axis 7: Grammar / Viewpoint Framing (原Axis 5)
# ============================================================
GRAMMAR = {
    "bare": "",
    "photo_of": "a photo of a",
    "aerial": "an aerial view of a",
    "satellite": "a top-down satellite image of a",
}

# ============================================================
# Axis 6: Negation (Phase 2 only — confusable pairs)
# ============================================================
CONFUSABLE_PAIRS = {
    "basketball_court": ["tennis court"],
    "tennis_court": ["basketball court"],
    "bridge": ["overpass"],
    "overpass": ["bridge"],
    "baseball_field": ["ground track field", "golf course"],
    "ground_track_field": ["baseball field", "stadium"],
    "stadium": ["ground track field"],
    "expressway_service_area": ["expressway toll station"],
    "expressway_toll_station": ["expressway service area"],
    "chimney": ["storage tank"],
    "storage_tank": ["chimney"],
}

# ============================================================
# DIOR 20 Classes
# ============================================================
DIOR_CLASSES = list(GRANULARITY.keys())

# ============================================================
# Axis Configuration
# ============================================================
AXIS_CONFIG = {
    "granularity": {
        "name": "Semantic Granularity",
        "baseline": "standard",
        "levels": ["coarse", "standard", "fine"],
        "perturbations": ["coarse", "fine"],
    },
    "spatial": {
        "name": "Spatial Context",
        "baseline": "none",
        "levels": ["none", "location", "context"],
        "perturbations": ["location", "context"],
    },
    "geometric": {
        "name": "Geometric Descriptor",
        "baseline": "none",
        "levels": ["none", "shape", "geometry"],
        "perturbations": ["shape", "geometry"],
    },
    "scale": {
        "name": "Scale Descriptor",
        "baseline": "none",
        "levels": ["none", "size", "comparative"],
        "perturbations": ["size", "comparative"],
    },
    "grammar": {
        "name": "Grammar/Viewpoint",
        "baseline": "bare",
        "levels": ["bare", "photo_of", "aerial", "satellite"],
        "perturbations": ["photo_of", "aerial", "satellite"],
    },
    "spectral": {
        "name": "Spectral/Color",
        "baseline": "none",
        "levels": ["none", "color", "spectral"],
        "perturbations": ["color", "spectral"],
    },
    "texture": {
        "name": "Texture/Pattern",
        "baseline": "none",
        "levels": ["none", "texture", "pattern"],
        "perturbations": ["texture", "pattern"],
    },
    "negation": {
        "name": "Negation",
        "baseline": "none",
        "levels": ["none", "negation"],
        "perturbations": ["negation"],
        "phase2_only": True,
    },
}

# ============================================================
# Prompt Assembly
# ============================================================
def _fix_article(prefix, noun_text):
    """Fix 'a airplane' -> 'an airplane' based on first word of noun."""
    if prefix.endswith(" a"):
        first_word = noun_text.split()[0].lower()
        if first_word[0] in "aeiou":
            return prefix[:-1] + "an"
    return prefix


def build_prompt(cls, granularity="standard", spatial="none", geometric="none",
                 scale="none", spectral="none", texture="none",
                 grammar="bare", negation="none"):
    """
    Assemble prompt: [grammar_prefix] [scale_adj] [noun_core] [, geometry] [, negation]
    """
    noun = GRANULARITY[cls][granularity]

    # Grammar prefix with a/an correction
    grammar_prefix = GRAMMAR[grammar]
    if grammar_prefix:
        grammar_prefix = _fix_article(grammar_prefix, noun)

    # Build core: noun + spectral + spatial + texture + geometry
    # Skip any descriptor text already contained in granularity noun
    core = noun
    if spectral != "none":
        s = SPECTRAL[cls].get(spectral, "")
        if s and s.lower() not in core.lower():
            core = core + ", " + s
    if spatial != "none":
        s = SPATIAL[cls].get(spatial, "")
        if s and s.lower() not in core.lower():
            core = core + " " + s
    if texture != "none":
        t = TEXTURE[cls].get(texture, "")
        if t and t.lower() not in core.lower():
            core = core + ", " + t
    if geometric != "none":
        g = GEOMETRIC[cls].get(geometric, "")
        if g and g.lower() not in core.lower():
            core = core + ", " + g

    # Negation
    if negation == "negation" and cls in CONFUSABLE_PAIRS:
        confusers = CONFUSABLE_PAIRS[cls]
        core = core + ", not a " + ", not a ".join(confusers)

    # Scale modifier
    scale_mod = SCALE[cls].get(scale, "") if scale != "none" else ""

    # Assemble
    parts = [p for p in [grammar_prefix, scale_mod, core] if p]
    return " ".join(parts).strip()


def build_baseline(cls):
    """Baseline prompt: standard granularity, all axes at default."""
    return build_prompt(cls)


def build_ofat_variants(cls):
    """
    Build all OFAT test variants for one class.
    Returns: {axis_name: {level: prompt_string}}
    """
    variants = {"baseline": build_baseline(cls)}
    for axis_name, config in AXIS_CONFIG.items():
        if config.get("phase2_only"):
            continue
        variants[axis_name] = {}
        for level in config["perturbations"]:
            kwargs = {a: "none" for a in ["spatial", "geometric", "scale", "spectral", "texture"]}
            kwargs["granularity"] = "standard"
            kwargs["grammar"] = "bare"
            kwargs[axis_name] = level
            variants[axis_name][level] = build_prompt(cls, **kwargs)
    return variants


# ============================================================
# Verification
# ============================================================
if __name__ == "__main__":
    test_classes = ["airplane", "ship", "storage_tank", "basketball_court", "stadium"]
    print("=" * 70)
    print("Prompt Axes Definition — Verification")
    print("=" * 70)

    for cls in test_classes:
        print(f"\n{'─'*70}")
        print(f"  [{cls.upper()}]")
        print(f"{'─'*70}")
        baseline = build_baseline(cls)
        print(f'  BASELINE: "{baseline}"')

        variants = build_ofat_variants(cls)
        for axis, levels in variants.items():
            if axis == "baseline":
                continue
            for level, prompt in levels.items():
                print(f'  {axis:12s} | {level:12s} | "{prompt}"')

        # Example combinatorial prompts
        print(f"\n  --- Combinatorial examples ---")
        examples = [
            ("fine+aerial", {"granularity": "fine", "grammar": "aerial"}),
            ("std+spatial+geom", {"granularity": "standard", "spatial": "context", "geometric": "shape"}),
            ("all best guess", {"granularity": "fine", "spatial": "context", "scale": "size", "grammar": "aerial"}),
        ]
        for label, kwargs in examples:
            p = build_prompt(cls, **kwargs)
            print(f'  {label:20s} "{p}"')

        if cls in CONFUSABLE_PAIRS:
            p_neg = build_prompt(cls, granularity="standard", negation="negation")
            print(f'  {"std+negation":20s} "{p_neg}"')

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_variants = 1 + sum(
        len(c["perturbations"]) for c in AXIS_CONFIG.values() if not c.get("phase2_only")
    )
    print(f"Classes: {len(DIOR_CLASSES)}")
    print(f"OFAT variants/class: {n_variants} (1 baseline + {n_variants-1} perturbations)")
    print(f"Total prompts: {len(DIOR_CLASSES) * n_variants}")
    print(f"Images/class: 20")
    print(f"Inferences/model: {len(DIOR_CLASSES) * n_variants * 20}")
