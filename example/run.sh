#!/bin/bash


###################
# UHA
###################
for target in neal brownian log_sonar
do
    for nbridges in 16 32 64
    do 
        for bs in 32
        do
            for lr  in {0.01,0.001,0.0001}
            do
                for eb in 0.25
                do
                    for id in {1..32}
                    do
                        python run_uha.py --target $target --nbridges $nbridges --bs $bs --lr $lr --id $id --epsbound $eb
                    done
                done
            done
        done
    done
done

wait
echo -e 'uha done'

