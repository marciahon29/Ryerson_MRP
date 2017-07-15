# This creates the different folds. There are 5 folds.
#
# The structure is as follows:
#
# data_00
#       > train
#		> alzheimers
#		> nonalzheimers
#	> validation
#		> alzheimers
#		> nonalzheimers
#
#
# data_##, goes from 00 to 04 (totalling 5 folds)
#
#
# In order to run, this file must be in a folder with:
#	alzheimers (folder)
#	nonalzheimers (folder)
#	5_folds.sh (script)
#
#
#
#
for seq_f in `seq 0 4`
do
	mkdir data_0$seq_f
	mkdir data_0$seq_f/train
	mkdir data_0$seq_f/train/alzheimers
	mkdir data_0$seq_f/train/nonalzheimers
	mkdir data_0$seq_f/validation
	mkdir data_0$seq_f/validation/alzheimers
	mkdir data_0$seq_f/validation/nonalzheimers
done


cd alzheimers/folds
for seq_f in `seq 0 4`
do
	for direc in `ls`
	do
		if [[ $direc =~ ${seq_f}_train ]]
		then
			cp $direc/*.jpg ../../data_0$seq_f/train/alzheimers
		fi
		
		if [[ $direc =~ ${seq_f}_validation ]]
		then
			cp $direc/*.jpg ../../data_0$seq_f/validation/alzheimers
		fi
    	done
done

cd ../..

cd nonalzheimers/folds
for seq_f in `seq 0 4`
do
	for direc in `ls`
	do
		if [[ $direc =~ ${seq_f}_train ]]
		then
			cp $direc/*.jpg ../../data_0$seq_f/train/nonalzheimers
		fi
		
		if [[ $direc =~ ${seq_f}_validation ]]
		then
			cp $direc/*.jpg ../../data_0$seq_f/validation/nonalzheimers
		fi
    	done
done

