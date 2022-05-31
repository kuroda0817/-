#/bin/bash
args1=(50000)
args2=(1000)
args3=("sports" "music" "eating" "travel")
for a1 in ${args1[@]};do
    for a2 in ${args2[@]};do
        for a3 in ${args3[@]};do
            python3 main_RL.py -A train --model sample${a1}_${a2}_${a3} --ep ${a1} --m_node ${a2} --learning_theme ${a3}
        done
    done
done
