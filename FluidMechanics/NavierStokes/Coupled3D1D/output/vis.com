$lastStep = 100;
$numProcs = 1;
$tinc = 5;

for ($i=0;$i<=$lastStep;$i=$i+$tinc) 
{
 for ($p=0;$p<$numProcs;$p=$p+1) 
  {
   $filename = "./TimeStep3D_".$i.".part".$p.".exnode"
   print "Reading $filename time $i\n";
   gfx read node "$filename" time $i;
  }
}

for ($p=0;$p<$numProcs;$p=$p+1) 
{
 gfx read element "./TimeStep3D_0.part".$p.".exelem";
}

gfx define faces egroup "3DIliac"
gfx def field /3DIliac/velocity3D component U.1 U.2 U.3;
gfx def field /3DIliac/pressure3D component U.4;
gfx def field /3DIliac/velmag3D magnitude field velocity3D;

for ($i=0;$i<=$lastStep;$i=$i+$tinc) 
  {
  for ($p=0;$p<$numProcs;$p=$p+1) 
    {
     $filename = "./TimeStep1D_".$i.".part".$p.".exnode"
     print "Reading $filename time $i\n";
     gfx read node "$filename" time $i;
    }
  }
for ($p=0;$p<$numProcs;$p=$p+1) 
  {
   gfx read element "./TimeStep1D_0.part".$p.".exelem";
  }

gfx def field /1D_LegArteries/flow1D component Flow_Area.1
gfx def field /1D_LegArteries/area1D component Flow_Area.2
gfx def field /1D_LegArteries/pressure1D component Pressure_Stress_Flow.1
gfx def field /1D_LegArteries/pi constant 3.141592653589793
gfx def field /1D_LegArteries/A0 constant 0.7853981633974484
gfx def field /1D_LegArteries/AOverPi divide_components fields area1D pi
gfx def field /1D_LegArteries/radius1D sqrt field AOverPi
gfx def field /1D_LegArteries/beta constant 1961.8752451900639
gfx def field /1D_LegArteries/secondTerm constant -1738.666666666666
gfx def field /1D_LegArteries/sqA sqrt field area1D
gfx def field /1D_LegArteries/firstTerm multiply_components fields sqA beta
gfx def field /1D_LegArteries/pOfA1D add fields firstTerm secondTerm

gfx mod g_e /3DIliac surfaces select_on material default data pressure3D spectrum default selected_material default_selected render_shaded;
gfx modify g_element /1D_LegArteries lines select_on material default data pressure1D spectrum default selected_material default_selected;
gfx modify g_element /3DIliac surfaces select_on material default data pressure3D spectrum default selected_material default_selected render_shaded;
gfx modify spectrum default autorange;

gfx create window 1;
gfx create time_editor;
gfx edit spectrum;
