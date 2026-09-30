NAME          TINYMILP
* max 5x + 4y  s.t.  6x + 4y <= 24,  x + 2y <= 6,  x, y >= 0 integer   (as min -5x - 4y)
* LP relaxation: x = 3, y = 1.5 (obj -21).  Integer optimum: x = 4, y = 0 (obj -20).
ROWS
 N  COST
 L  C1
 L  C2
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST      -5.0      C1        6.0
    X         C2        1.0
    Y         COST      -4.0      C1        4.0
    Y         C2        2.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       C1        24.0      C2        6.0
BOUNDS
 UP BND       X         10.0
 UP BND       Y         10.0
ENDATA
