# This script creates the 5-folds for Alzheimers
#
# Total Alzheimers: 3200, Total NonAlzheimers: 3200
# Total images: 6400
#
# The train/validation division is 80% to 20%
# Meaning test is 2560 images and validation is 640 images for each category
#       * 2560 (train - alzheimers) and 2560 (train - nonalzheiemrs)
#       * 640 (validation - nonalzheimers) and 640 (validation - alzheimers)
#
# This is for 5-fold, therefore, we are creating 5 folders each having 560 images.
# These folders are named: fold_0# and fold_0#, where # is from 0 to 9.
#
# After these 5 folders are done, we join 4 and keep one out for a total of 5 times.
#
#

# Extract all alzheimer's images from folders
# Put all results into the "alzheimers" folder
# rename the images
mkdir nonalzheimers
seq=0
for file_name in */*
do
	seq=$((seq+1))
	foo=$(printf "%04d" $seq)
	cp $file_name nonalzheimers/NAL${foo}.jpg
done



# To create the five folders labelled from fold_00 to fold_04
# Each folder contains 640 images
cd nonalzheimers
mkdir folds
for seq_f in `seq 0 4`
do
    mkdir folds/fold_0$seq_f

    mv `(ls -I "folds" | sort -R | head -640)` folds/fold_0$seq_f

    cd folds/fold_0$seq_f
    seq0=0
    for file_name in *.*
    do
        seq0=$((seq0+1))
        foo=$(printf "%04d" $seq0)
        mv $file_name fold_0${seq_f}_NAL${foo}.jpg
    done
    cd ../..
done


# Creating first of 5-fold
# Done inside the folds
cd folds
for seq_f in `seq 0 4`
do
    # Create the validation set for a given fold
    # Use only one fold
    cp -rf ./fold_0${seq_f} ./fold_0${seq_f}_validation
    # rename the files in the validation folder
    cd fold_0${seq_f}_validation
    seq0=0
    for file_name in *.*
    do
        seq0=$((seq0+1))
        foo=$(printf "%04d" $seq0)
        mv $file_name NAL${foo}.jpg
    done
    cd ..

    # Create the training set for the given fold
    # This is achieved by adding all the other non-validation folds together
    mkdir fold_0${seq_f}_train
    for seqff in `seq 0 4`
    do
	if [ "$seqff" -ne "$seq_f" ]
	then
		cp ./fold_0$seqff/*.jpg ./fold_0${seq_f}_train
	fi
    done

    # rename all files in the training folder
    cd fold_0${seq_f}_train
    seq0=0
    for file_name in *.*
    do
        seq0=$((seq0+1))
        foo=$(printf "%04d" $seq0)
        mv $file_name NAL${foo}.jpg
    done

    cd ..
done

	
	

