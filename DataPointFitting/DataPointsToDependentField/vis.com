gfx create region dataFit;
gfx read region DataFitExample.xml region dataFit;
gfx define face egroup dataFit;

gfx modify g_element dataFit general clear;
gfx modify g_element dataFit lines coordinate dataFit.geometric select_on material default selected_material default_selected;
gfx modify g_element dataFit node_points as node_spheres glyph arrow_solid general size "0.001*0.001*0.001" centre 0,0,0 font default coordinate dataFit.geometric orientation dataFit.dependent scale_factors "0.1*0.05*0.05" select_on material default;
gfx modify spectrum default autorange;

gfx create region dataPoints;
gfx read region DataPoints.xml region dataPoints;
gfx modify g_element dataPoints node_points as node_spheres glyph cross general size "0.02*0.02*0.02" centre 0,0,0 font default coordinate dataPoints.geometric select_on material green;

gfx cre window 1;
