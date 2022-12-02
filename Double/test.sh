while read line
do
	pip install $line
done < requirements.freezed
