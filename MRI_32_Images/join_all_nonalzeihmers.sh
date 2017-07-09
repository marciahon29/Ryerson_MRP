mkdir nonalzheimers
seq=0
for file_name in */*
do
	seq=$((seq+1))
	foo=$(printf "%04d" $seq)
	cp $file_name nonalzheimers/NAL${foo}.jpg
done
