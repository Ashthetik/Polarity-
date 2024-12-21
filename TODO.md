## `./handler/visual/emotions/ned.cpp:31`:  [2] (misc) open:
Check when opening files - can an attacker redirect it (via symlinks), 
force the opening of special file type (e.g., device files), 
move things around to create a race condition, control its ancestors, or change its contents? (CWE-362).

## `./handler/visual/emotions/ned.cpp:37`:  [1] (buffer) read:
Check buffer boundaries if used in a loop including recursive loops (CWE-120, CWE-20).

# ANALYSIS SUMMARY:
```
Hits = 2
Lines analyzed = 855 in approximately 0.01 seconds (138627 lines/second)
Physical Source Lines of Code (SLOC) = 648
Hits@level = [0]   0 [1]   1 [2]   1 [3]   0 [4]   0 [5]   0
Hits@level+ = [0+]   2 [1+]   2 [2+]   1 [3+]   0 [4+]   0 [5+]   0
Hits/KSLOC@level+ = [0+] 3.08642 [1+] 3.08642 [2+] 1.54321 [3+]   0 [4+]   0 [5+]   0
Minimum risk level = 1
```