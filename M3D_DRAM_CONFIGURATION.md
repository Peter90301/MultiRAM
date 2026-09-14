# MultiRAM M3D DRAM Configuration

## Organization

| Parameter | Value |
| --- | ---: |
| Total capacity | 32 GB |
| Tiers | 8 |
| Capacity per tier | 4 GB |
| Physical layers | 1024 |
| Physical layers per tier | 128 |
| Channels | 32 |
| PUs per channel | 1 |
| Total channel-level PUs | 32 |
| Bank groups per channel | 2 |
| Banks per bank group | 4 |
| Banks per channel | 8 |
| Total banks | 256 |
| Capacity per bank | 1 Gb |

## Capacity Check

```text
8 tiers x 4 GB/tier = 32 GB

32 channels x 8 banks/channel x 1 Gb/bank
= 256 Gb
= 32 GB
```

The 1 Gb value is the stack-level capacity of each logical bank. Across eight
tiers, each bank contributes 128 Mb per tier. Treating every tier-local bank
slice as 1 Gb would instead imply 256 GB and would be inconsistent with the
specified 32 GB total capacity.

## PIM Partition

The existing functional split is retained and now explicitly tied to the 32
channel-level PUs:

| Function | Channels/PUs | Logical tiers |
| --- | ---: | ---: |
| Genomic alignment compute | 24 | 6 |
| Seeding/search | 8 | 2 |
| Total | 32 | 8 |

The 16 logical PEs inside each PU remain unchanged. They are logic-die compute
lanes and are not DRAM banks.

## Implemented Files

- `GenDP/GenDRAM/config_unified.py`
- `GenDP/GenDRAM/simulator_unified.py`
- `GenDP/GenDRAM/main_analysis.py`
- `GenDP2/GenDRAM/config_unified.py`
- `GenDP2/GenDRAM/simulator_unified.py`
- `GenDP2/GenDRAM/main_analysis.py`

Both configurations validate at import time that the tier-based and bank-based
capacity calculations equal 32 GB and that the search and compute PU counts
sum to one PU per channel.
