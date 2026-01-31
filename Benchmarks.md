
########## optis F ###########

[100%] Built target llama-server
✅ Fertig! Caching wurde genutzt. Speed: 1150 t/s Ready.
pat@ubun24:/opt/llama-mi50.cpp$ 
pat@ubun24:/opt/llama-mi50.cpp$ ./build/bin/llama-bench -m /home/pat/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |           pp512 |        812.92 ± 3.22 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |          pp1024 |       1042.52 ± 1.75 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |          pp2048 |       1160.78 ± 2.79 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |           tg128 |         75.96 ± 0.09 |


######### otiginal #########

build: 36ba1c2b (7881)
pat@ubun24:/opt/llama-mi50.cpp$ 
pat@ubun24:/opt/llama.cpp$ ./build/bin/llama-bench -m /home/pat/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |           pp512 |        809.02 ± 3.49 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |          pp1024 |       1035.85 ± 2.50 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |          pp2048 |       1145.63 ± 2.06 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm       | 999 |     2048 |  1 |           tg128 |         68.35 ± 0.18 |

build: 4fdbc1e4d (7875)
pat@ubun24:/opt/llama.cpp$ 


#### optis F+ C  ######
./build/bin/llama-bench -m /home/pat/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
load_backend: loaded ROCm backend from /opt/llama-mi50.cpp/build/bin/libggml-hip.so
load_backend: loaded CPU backend from /opt/llama-mi50.cpp/build/bin/libggml-cpu.so
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           pp512 |        825.51 ± 3.49 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp1024 |       1060.93 ± 2.04 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp2048 |       1184.59 ± 2.44 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           tg128 |         76.52 ± 0.05 |

build: 375a6791 (7885)

##### F + C + E ###
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           pp512 |        825.96 ± 3.54 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp1024 |       1063.20 ± 2.02 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp2048 |       1185.57 ± 2.64 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           tg128 |         76.10 ± 0.09 |

build: 5bd14c86 (7887)
pat@ubun24:/opt/llama-mi50.cpp$ 


####  F + C + E + G ###
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           pp512 |        827.05 ± 4.34 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp1024 |       1062.36 ± 1.78 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp2048 |       1183.81 ± 3.06 |
| qwen3moe 30B.A3B Q5_K - Medium |  20.24 GiB |    30.53 B | ROCm,ROCm  | 999 |     2048 |  1 |           tg128 |         76.46 ± 0.05 |


---
t@ubun24:/opt/llama.cpp$ ./bin/llama-bench -m /home/pat/models/Qwen3-8B-DeepSeek-v3.2-Speciale-Distill.q8_0.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
bash: ./bin/llama-bench: No such file or directory
pat@ubun24:/opt/llama.cpp$ ./build/bin/llama-bench -m /home/pat/models/Qwen3-8B-DeepSeek-v3.2-Speciale-Distill.q8_0.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm       | 999 |     2048 |  1 |           pp512 |        513.89 ± 0.32 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm       | 999 |     2048 |  1 |          pp1024 |        544.52 ± 0.42 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm       | 999 |     2048 |  1 |          pp2048 |        536.50 ± 0.83 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm       | 999 |     2048 |  1 |           tg128 |         59.53 ± 0.08 |

build: c3b87cebf (7885)

---
pat@ubun24:/opt/llama-mi50.cpp$ ./build/bin/llama-bench -m /home/pat/models/Qwen3-8B-DeepSeek-v3.2-Speciale-Distill.q8_0.gguf -p 512,1024,2048 -n 128 -ngl 999 -fa 1 -b 2048 -ub 2048
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
load_backend: loaded ROCm backend from /opt/llama-mi50.cpp/build/bin/libggml-hip.so
load_backend: loaded CPU backend from /opt/llama-mi50.cpp/build/bin/libggml-cpu.so
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |           pp512 |        718.80 ± 0.57 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp1024 |        752.89 ± 0.38 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp2048 |        734.37 ± 2.26 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |           tg128 |         64.22 ± 0.03 |

build: 4dcb6f81 (7893)

---
| model                          |       size |     params | backend    | ngl | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -: | --------------: | -------------------: |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |           pp512 |        724.57 ± 0.35 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp1024 |        757.46 ± 1.10 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |          pp2048 |        737.48 ± 2.35 |
| qwen3 8B Q8_0                  |   8.11 GiB |     8.19 B | ROCm,ROCm  | 999 |     2048 |  1 |           tg128 |         65.61 ± 0.03 |

build: 49dac82c (7933)
