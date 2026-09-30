NAME          TINY
* max 3x + 5y  s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x + y >= 1, x - z = 0 ; as min -3x-5y
ROWS
 N  COST
 L  LIM1
 L  LIM2
 L  MYEQN
 G  LOW
 E  LINK
COLUMNS
    X         COST      -3.0      LIM1      1.0
    X         MYEQN     3.0       LOW       1.0
    X         LINK      1.0
    Y         COST      -5.0      LIM2      2.0
    Y         MYEQN     2.0       LOW       1.0
    Z         LINK      -1.0
RHS
    RHS       LIM1      4.0       LIM2      12.0
    RHS       MYEQN     18.0      LOW       1.0
BOUNDS
 UP BND       Z         10.0
ENDATA
