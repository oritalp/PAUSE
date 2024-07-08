for epsilon_bar in 100
do
    for delta_f in 0.003
    do 
        python -u main.py --epsilon_bar $epsilon_bar --method_choosing_users ALSA --global_epochs 2 --delta_f $delta_f & \
        python -u main.py --epsilon_bar $epsilon_bar --method_choosing_users all users --global_epochs 2 --delta_f $delta_f & \
    done
done


