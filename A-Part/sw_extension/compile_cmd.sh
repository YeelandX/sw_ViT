mkdir -p build/temp.linux-sw_64-3.6
mkdir -p build/lib.linux-sw_64-3.6
swgcc -dumpfullversion -dumpversion
swgcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -mieee -fPIC -I./swops/ -I/home/export/online1/mdt00/shisuan/${USER}/.local/lib/python3.6/site-packages/torch/include -I/home/export/online1/mdt00/shisuan/${USER}/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/home/export/online1/mdt00/shisuan/${USER}/.local/lib/python3.6/site-packages/torch/include/TH -I/home/export/online1/mdt00/shisuan/${USER}/.local/lib/python3.6/site-packages/torch/include/THC -I/usr/sw/swpython/include/python3.6m -c swextension.cpp -o build/temp.linux-sw_64-3.6/swextension.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=swextension -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14
swgcc -mdynamic -shared build/temp.linux-sw_64-3.6/swextension.o -L/usr/sw/swpython/lib -L./swops/ -L/home/export/online1/mdt00/shisuan/${USER}/.local/lib/python3.6/site-packages/torch/lib -L/home/swpython/python3.6/python3_6_sw/swpython/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -lpython3.6m -o build/lib.linux-sw_64-3.6/swextension.cpython-36m.so -lswops -lm -lm_slave
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
swgcc -dumpfullversion -dumpversion
