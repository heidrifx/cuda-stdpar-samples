#!/bin/bash
~/fd/fd --max-depth=1 -t f --threads=1 -x bash -c "for i in {0..10}; do {}; done | ~/rg/rg -oe '(\d+)ms' > data/{.}.csv" \;
sed -i 's/ms//' data/*
~/fd/fd -x sed -i '1s/^/{/.}\n/' {} \; . data/
