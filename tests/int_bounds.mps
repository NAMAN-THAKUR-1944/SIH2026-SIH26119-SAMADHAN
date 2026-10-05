NAME          INTBOUNDS
ROWS
 N  COST
 L  R1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    XNONE     COST      1.0        R1        1.0
    XLO       COST      1.0        R1        1.0
    XUP       COST      1.0        R1        1.0
    XLONEG    COST      1.0        R1        1.0
    XMI       COST      1.0        R1        1.0
    MARKER                 'MARKER'                 'INTEND'
    YCONT     COST      1.0        R1        1.0
RHS
    RHS       R1        10.0
BOUNDS
 LO BND       XLO       2.0
 UP BND       XUP       5.0
 LO BND       XLONEG    -1.0
 MI BND       XMI
ENDATA
