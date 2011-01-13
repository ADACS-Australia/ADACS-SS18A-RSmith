#!/bin/sh

c=1

awk '{print $2,$3}' ../S5-H1segments-cat1.txt > S5h1.seg
awk '{print $2,$3}' ../S5-H2segments-cat1.txt > S5h2.seg
awk '{print $2,$3}' ../S5-L1segments-cat1.txt > S5l1.seg
awk '{print $2,$3}' ../S5-V1segments-cat1.txt > S5v1.seg
awk '{print $2,$3}' ../S6-H1segments-cat1.txt > S6h1.seg
awk '{print $2,$3}' ../S6-L1segments-cat1.txt > S6l1.seg
awk '{print $2,$3}' ../S6-V1segments-cat1.txt > S6v1.seg

while read line; do
    chunk_start=`echo $line | awk '{print $2}'`
    chunk_end=`echo $line | awk '{print $3}'`
    if [ "$chunk_start" = "" ]; then continue; fi
    echo "Building segment lists for chunk ${c}..."
    echo $chunk_start $chunk_end > ./tmp
    h1_lt=0; h2_lt=0; l1_lt=0; v1_lt=0; 

    # S5 case
    if [ $chunk_end -lt 920000000 ]; then
	segexpr 'intersection(tmp,S5h1.seg)' -include index,duration > ./S5-H1segments-cat1_c${c}.txt
	segexpr 'intersection(tmp,S5h2.seg)' -include index,duration > ./S5-H2segments-cat1_c${c}.txt
	segexpr 'intersection(tmp,S5l1.seg)' -include index,duration > ./S5-L1segments-cat1_c${c}.txt
	segexpr 'intersection(tmp,S5v1.seg)' -include index,duration > ./S5-V1segments-cat1_c${c}.txt
        
	h1_lt=`segsum ./S5-H1segments-cat1_c${c}.txt`
	h2_lt=`segsum ./S5-H2segments-cat1_c${c}.txt`
	l1_lt=`segsum ./S5-L1segments-cat1_c${c}.txt`
	v1_lt=`segsum ./S5-V1segments-cat1_c${c}.txt`

    # S6 case
    else
	segexpr 'intersection(tmp,S6h1.seg)' -include index,duration > ./S6-H1segments-cat1_c${c}.txt
	segexpr 'intersection(tmp,S6l1.seg)' -include index,duration > ./S6-L1segments-cat1_c${c}.txt
	segexpr 'intersection(tmp,S6v1.seg)' -include index,duration > ./S6-V1segments-cat1_c${c}.txt
        
	h1_lt=`segsum ./S6-H1segments-cat1_c${c}.txt`
	l1_lt=`segsum ./S6-L1segments-cat1_c${c}.txt`
	v1_lt=`segsum ./S6-V1segments-cat1_c${c}.txt`
    fi

    echo "H1 livetime=${h1_lt} sec"
    echo "H2 livetime=${h2_lt} sec"
    echo "L1 livetime=${l1_lt} sec"
    echo "V1 livetime=${v1_lt} sec"

    let "c+=1"
done < chunks.txt

rm -f ./tmp

exit 0
