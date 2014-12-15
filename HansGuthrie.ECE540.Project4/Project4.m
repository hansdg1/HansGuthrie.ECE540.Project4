%generate graph for ki 0, 0, 0 and kp 4, 8, 12
clear
close all

load( 'ki_0_and_kp_4.csv' );
load( 'ki_0_and_kp_8.csv' );
load( 'ki_0_and_kp_12.csv' );

figure(1);

plot( ki_0_and_kp_4(:,1), ki_0_and_kp_4(:,2), 'b' );
hold on
plot( ki_0_and_kp_8(:,1), ki_0_and_kp_8(:,2), 'r' );
hold on
plot( ki_0_and_kp_12(:,1), ki_0_and_kp_12(:,2), 'g' );
hold on
title( 'Ki = 0, Varying Kp', 'fontweight','bold' );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Kp = 4', 'Kp = 8', 'Kp = 12' );
print -dpng Fig01.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%and so on for all 6 graphs
clear
close all

load( 'ki_2_and_kp_4.csv' );
load( 'ki_2_and_kp_8.csv' );
load( 'ki_2_and_kp_12.csv' );

figure(2);

plot( ki_2_and_kp_4(:,1), ki_2_and_kp_4(:,2), 'b' );
hold on
plot( ki_2_and_kp_8(:,1), ki_2_and_kp_8(:,2), 'r' );
hold on
plot( ki_2_and_kp_12(:,1), ki_2_and_kp_12(:,2), 'g' );
hold on
title( 'Ki = 2, Varying Kp', 'fontweight','bold'  );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Kp = 4', 'Kp = 8', 'Kp = 12' );
print -dpng Fig02.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

load( 'ki_4_and_kp_4.csv' );
load( 'ki_4_and_kp_8.csv' );
load( 'ki_4_and_kp_12.csv' );

figure(3);

plot( ki_4_and_kp_4(:,1), ki_4_and_kp_4(:,2), 'b' );
hold on
plot( ki_4_and_kp_8(:,1), ki_4_and_kp_8(:,2), 'r' );
hold on
plot( ki_4_and_kp_12(:,1), ki_4_and_kp_12(:,2), 'g' );
hold on
title( 'Ki = 4, Varying Kp', 'fontweight','bold'  );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Kp = 4', 'Kp = 8', 'Kp = 12' );
print -dpng Fig03.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

load( 'ki_0_and_kp_4.csv' );
load( 'ki_2_and_kp_4.csv' );
load( 'ki_4_and_kp_4.csv' );

figure(4);

plot( ki_0_and_kp_4(:,1), ki_0_and_kp_4(:,2), 'b' );
hold on
plot( ki_2_and_kp_4(:,1), ki_2_and_kp_4(:,2), 'r' );
hold on
plot( ki_4_and_kp_4(:,1), ki_4_and_kp_4(:,2), 'g' );
hold on
title( 'Kp = 4, Varying Ki', 'fontweight','bold'  );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Ki = 0', 'Ki = 2', 'Ki = 4' );
print -dpng Fig04.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

load( 'ki_0_and_kp_8.csv' );
load( 'ki_2_and_kp_8.csv' );
load( 'ki_4_and_kp_8.csv' );

figure(5);

plot( ki_0_and_kp_8(:,1), ki_0_and_kp_8(:,2), 'b' );
hold on
plot( ki_2_and_kp_8(:,1), ki_2_and_kp_8(:,2), 'r' );
hold on
plot( ki_4_and_kp_8(:,1), ki_4_and_kp_8(:,2), 'g' );
hold on
title( 'Kp = 8, Varying Ki', 'fontweight','bold'  );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Ki = 0', 'Ki = 2', 'Ki = 4' );
print -dpng Fig05.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

load( 'ki_0_and_kp_12.csv' );
load( 'ki_2_and_kp_12.csv' );
load( 'ki_4_and_kp_12.csv' );

figure(6);

plot( ki_0_and_kp_12(:,1), ki_0_and_kp_12(:,2), 'b' );
hold on
plot( ki_2_and_kp_12(:,1), ki_2_and_kp_12(:,2), 'r' );
hold on
plot( ki_4_and_kp_12(:,1), ki_4_and_kp_12(:,2), 'g' );
hold on
title( 'Kp = 12, Varying Ki', 'fontweight','bold'  );
xlabel( 'Time (Seconds)', 'fontweight','bold'  );
ylabel( 'Output (Volts)', 'fontweight','bold'  );
grid
legend( 'Ki = 0', 'Ki = 2', 'Ki = 4' );
print -dpng Fig06.png
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
