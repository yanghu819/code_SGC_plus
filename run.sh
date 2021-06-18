
dataset="reddit"
orderset="2 3 4 5 6"
lrset="0.01 0.001"
wdset="5e-4 5e-5"
epochset="100"
tmpset="1.0"
neibor_set="1.0"
plus_set="1 0"
for data in $dataset
do
    for order in $orderset
    do
        for lr in $lrset
        do
            for wd in $wdset
            do
                for epoch in $epochset
                do
                    for tmp in $tmpset
                    do
                        for neibor in $neibor_set
                        do
                            for plus in $plus_set
                            do 
                            python3 train.py --lr=$lr --weight_decay=$wd --data=$data --order=$order --epochs=$epoch --tmp=$tmp --neibor_ratio=$neibor --plus=$plus
                            done
                        done
                    done
                    
                done
            done
        done
    done
done


