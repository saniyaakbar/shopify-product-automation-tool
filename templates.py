"""
templates.py

Defines prompt templates for image generation.

Each template string should contain `{plant_name}` which will be replaced
with the actual plant name from the CSV.
"""

TEMPLATES = {
   "studio": (
    "High-end luxury product photography of a real potted {plant_name} placed in front of a warm cream-beige fabric curtain backdrop with soft, natural vertical folds (elegant, refined, not wrinkled or messy). "
    "Natural late-afternoon daylight entering softly from the left side, creating gentle sculpted shadows across the curtain and subtle dimensional highlights on the leaves. "
    "Visible smooth light-to-shadow gradient across the fabric background for depth and tonal richness (not flat, not evenly lit). "

    "Balanced contrast with clean highlights and natural shadow falloff. Slightly wider framing with generous negative space around the plant for a premium editorial composition. "
    "Foreground sharply focused with realistic lens falloff (85mm lens), soft depth separation between subject and backdrop. "

    "True botanical color accuracy with subtle leaf gloss variation, realistic organic imperfections, and natural growth pattern (no artificial symmetry). "
    "Premium glossy white ceramic planter with refined curvature, elegant edge highlights, and soft reflective sheen (bright clean white, not matte, not plastic). "

    "Shot on full-frame camera, 85mm lens, natural depth of field, realistic exposure. "
    "Refined luxury plant photography aesthetic with calm Scandinavian editorial mood. "
    "No CGI look, no artificial texture, no flat lighting, no harsh dramatic shadows, no beige planter, no text, no watermark."
),

    "lifestyle":  (
    "High-end minimal macro photography of a real {plant_name} leaf or flower "
    "in a perfect square composition (2400 × 2400 framing). "

    "Tight close-up shot focusing on natural surface texture, delicate vein structure, "
    "subtle tonal variation, and realistic organic micro-imperfections "
    "(no artificial symmetry, no overly perfect edges). "

    "Soft diffused daylight entering gently from the left side, creating smooth highlight "
    "roll-off and refined shadow gradients across the surface "
    "(not harsh, not dramatic, not flat). "

    "Background in warm neutral beige tones, softly blurred with true optical depth of field "
    "(100mm macro full-frame lens look, natural bokeh, not artificial blur). "
    "Subtle tonal gradient in the background for dimensional richness. "

    "True botanical color accuracy with rich yet realistic tones, no oversaturation, "
    "no exaggerated greens or reds, no artificial gloss. "

    "Premium botanical editorial aesthetic, refined Scandinavian luxury mood. "
    "Photographic realism only — no CGI rendering look, no artificial smoothness, "
    "no excessive sharpening, no text, no watermark."
),

   "macro":(
    "High-end minimal studio product photography of a real potted {plant_name} "
    "in a perfect square composition (2400 × 2400 framing). "

    "Soft cream-beige fabric curtain backdrop with elegant natural vertical folds, "
    "refined and smooth (not wrinkled, not messy). The curtain features a subtle "
    "light-to-shadow gradient for depth and dimensional richness "
    "(not flat, not evenly lit). "

    "Warm diffused daylight entering softly from the left side, creating gentle "
    "sculpted shadows with smooth edge transitions across the curtain and a soft "
    "natural shadow beneath the planter. Balanced cinematic contrast with clean "
    "highlights and realistic shadow depth. "

    "The plant is centered on a clean matte white surface with generous negative "
    "space around it for a premium ecommerce presentation. Organic natural growth "
    "pattern with slight leaf height variation and realistic micro-imperfections "
    "(no artificial symmetry, no perfect alignment). True botanical color accuracy "
    "with subtle natural leaf gloss variation. "

    "Premium glossy white ceramic planter with smooth refined curvature, elegant "
    "specular highlights, and soft reflective sheen (bright clean white, not matte, not plastic). "

    "In the background, softly blurred with natural depth of field (85mm full-frame lens look), "
    "a small round light-wood side table with white legs holds a clear glass jug "
    "and a small glass of water. The props are subtle, understated, and do not visually "
    "compete with the plant. "

    "Refined Scandinavian luxury aesthetic, modern indoor plant webshop styling. "
    "Calm, airy, elevated mood. "

    "No CGI appearance, no artificial smoothness, no harsh dramatic lighting, "
    "no flat background, no beige planter, no clutter, no text, no watermark."
),
"luxury_livingroom": (
        "Ultra-realistic luxury interior editorial photography of a real healthy potted {plant_name}, "
        "natural organic growth structure with realistic botanical variation, positioned slightly left of frame center "
        "inside a modern minimalist Scandinavian living room corner. "

        "The plant is placed in fine light beige decorative pebbles inside a large smooth rounded white ceramic planter "
        "with spherical bowl shape and refined curvature producing soft natural reflections. "
        "Planter height approximately one-third of total plant height. "

        "Background consists of warm neutral beige plaster walls meeting at a corner intersection with subtle texture "
        "and natural tonal variation. No artwork or decorations. Thin white baseboard along the bottom edge. "

        "Strong warm late-afternoon sunlight entering from frame left at approximately 35–45 degree angle, "
        "filtered through sheer off-white linen curtains creating soft diffused illumination and crisp elongated "
        "plant shadows projected onto the right wall with clearly defined leaf silhouettes. "

        "Floor-length sheer linen curtains visible only on extreme left edge with soft vertical folds "
        "and glowing translucent fabric diffusion. Window not visible. "

        "Light natural oak wooden flooring with matte finish and realistic horizontal grain texture. "

        "Foreground left contains a partially visible modern light-beige upholstered sofa armrest slightly out of focus. "
        "Neutral cream rectangular rug edge partially visible beneath planter. "

        "Right side includes a small Scandinavian wooden side table with round top and tapered legs in light natural wood tone, "
        "styled with a matte beige ceramic vase holding dried stems, a transparent cylindrical drinking glass half-filled with water, "
        "and two stacked neutral hardcover books. "

        "Eye-level interior photography with straight perspective and slight three-quarter viewing angle toward corner. "
        "Shot on full-frame camera using 50mm lens at f/4 aperture with moderate depth of field. "

        "Warm neutral color grading consisting of cream, beige, soft wood tones and natural greens. "
        "Calm airy premium residential styling with high-end interior magazine aesthetic. "

        "Photorealistic rendering with physically accurate natural lighting, realistic global illumination, "
        "correct shadow physics, ultra-detailed textures and realistic reflections. "
        "No CGI appearance, no stylization, no text, no watermark."
    ),
    "macro_hydrated": (
    "Ultra-realistic botanical macro photography of the upper crown of a healthy {plant_name}, "
    "tightly framed close-up showing overlapping lance-shaped leaves emerging naturally from the central growth point. "

    "Leaves appear smooth, mature, and organically layered with realistic botanical variation, "
    "natural curvature, subtle asymmetry, and authentic growth structure. True plant texture visible "
    "with fine surface detail and natural tonal transitions. "

    "Leaf surfaces contain fresh natural water droplets distributed irregularly across multiple leaves, "
    "including micro condensation beads, medium rounded droplets, and subtle droplet merging. "
    "Accurate surface tension behavior with realistic light refraction and magnification through droplets, "
    "and soft specular highlights reflecting incoming natural light. Droplets follow natural leaf vein direction "
    "without artificial spray patterns. "

    "Soft warm diffused daylight entering gently from the upper left direction, producing smooth highlight roll-off "
    "along leaf ridges and gradual shadow transitions between layered foliage. "
    "No harsh reflections and no artificial shine. "

    "Background consists of softly blurred cream-beige sheer fabric curtains, not a solid wall, "
    "creating an elegant textile backdrop with extremely smooth optical bokeh and gentle vertical tonal gradients. "
    "Curtain folds remain subtle and atmospheric rather than visually dominant. "

    "Very shallow macro depth of field with foreground leaves razor sharp while rear foliage "
    "falls naturally into optical blur created by real lens compression rather than artificial background blur. "

    "Professional botanical macro photography using full-frame camera simulation with 100mm macro lens "
    "at approximately f/3.5 aperture, producing natural focus falloff and refined subject separation. "

    "Square composition focused on plant crown only, no planter visible, no exposed soil, "
    "tight editorial crop emphasizing freshness, hydration, and leaf texture detail. "

    "True botanical color accuracy with rich natural greens and realistic tonal variation, "
    "no oversaturation and no artificial gloss enhancement. "

    "Premium Scandinavian botanical editorial aesthetic with calm luxury indoor plant styling. "
    "Photorealistic rendering with physically accurate lighting, realistic moisture physics, "
    "ultra-detailed textures and no CGI appearance, no illustration look, no text, no watermark."
),

}
