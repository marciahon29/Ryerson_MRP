mkdir train
mkdir validation

mv `(ls | sort -R | head -320)` validation
mv *.jpg train

cd train
seq=0
for file_name in *
do
	seq=$((seq+1))
	foo=$(printf "%04d" $seq)
	mv $file_name NAL${foo}.jpg
done

cd ..
 
cd validation

seq=0
for file_name in *
do
	seq=$((seq+1))
	foo=$(printf "%04d" $seq)
	mv $file_name NAL${foo}.jpg
done

