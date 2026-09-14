# Dense Flight Multi-CI Shell Audit

## Flight 52
D_multi(52) = 21

### l=1
shell_1(52) = [77]
D_multi(v)-1 values = [{"flight_id": 77, "multi_degree": 22, "degree_minus_one": 21}]
sum(D_multi(v)-1) = 21
CI_1(52) = (21 - 1) * 21 = 420.0

### l=2
shell_2(52) = [76]
D_multi(v)-1 values = [{"flight_id": 76, "multi_degree": 1, "degree_minus_one": 0}]
sum(D_multi(v)-1) = 0
CI_2(52) = (21 - 1) * 0 = 0.0

### l=3
shell_3(52) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_3(52) = (21 - 1) * 0 = 0.0

### l=4
shell_4(52) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_4(52) = (21 - 1) * 0 = 0.0

## Flight 77
D_multi(77) = 22

### l=1
shell_1(77) = [52, 76]
D_multi(v)-1 values = [{"flight_id": 52, "multi_degree": 21, "degree_minus_one": 20}, {"flight_id": 76, "multi_degree": 1, "degree_minus_one": 0}]
sum(D_multi(v)-1) = 20
CI_1(77) = (22 - 1) * 20 = 420.0

### l=2
shell_2(77) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_2(77) = (22 - 1) * 0 = 0.0

### l=3
shell_3(77) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_3(77) = (22 - 1) * 0 = 0.0

### l=4
shell_4(77) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_4(77) = (22 - 1) * 0 = 0.0

## Flight 70
D_multi(70) = 15

### l=1
shell_1(70) = [87, 91]
D_multi(v)-1 values = [{"flight_id": 87, "multi_degree": 15, "degree_minus_one": 14}, {"flight_id": 91, "multi_degree": 1, "degree_minus_one": 0}]
sum(D_multi(v)-1) = 14
CI_1(70) = (15 - 1) * 14 = 196.0

### l=2
shell_2(70) = [88]
D_multi(v)-1 values = [{"flight_id": 88, "multi_degree": 2, "degree_minus_one": 1}]
sum(D_multi(v)-1) = 1
CI_2(70) = (15 - 1) * 1 = 14.0

### l=3
shell_3(70) = [61]
D_multi(v)-1 values = [{"flight_id": 61, "multi_degree": 1, "degree_minus_one": 0}]
sum(D_multi(v)-1) = 0
CI_3(70) = (15 - 1) * 0 = 0.0

### l=4
shell_4(70) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_4(70) = (15 - 1) * 0 = 0.0

## Flight 87
D_multi(87) = 15

### l=1
shell_1(87) = [70, 88]
D_multi(v)-1 values = [{"flight_id": 70, "multi_degree": 15, "degree_minus_one": 14}, {"flight_id": 88, "multi_degree": 2, "degree_minus_one": 1}]
sum(D_multi(v)-1) = 15
CI_1(87) = (15 - 1) * 15 = 210.0

### l=2
shell_2(87) = [61, 91]
D_multi(v)-1 values = [{"flight_id": 61, "multi_degree": 1, "degree_minus_one": 0}, {"flight_id": 91, "multi_degree": 1, "degree_minus_one": 0}]
sum(D_multi(v)-1) = 0
CI_2(87) = (15 - 1) * 0 = 0.0

### l=3
shell_3(87) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_3(87) = (15 - 1) * 0 = 0.0

### l=4
shell_4(87) = []
D_multi(v)-1 values = []
sum(D_multi(v)-1) = 0
CI_4(87) = (15 - 1) * 0 = 0.0
