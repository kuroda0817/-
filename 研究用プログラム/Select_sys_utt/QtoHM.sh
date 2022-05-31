#/bin/bash
cp $1/$1_Q .
python3 pickle2.py $1 409 33
python3 sort.py $1
python3 heat.py $1


