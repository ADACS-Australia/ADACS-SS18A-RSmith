if ligolw_add --verbose inspiral_event_id_test_in?.xml* | diff -q -s - inspiral_event_id_test_out.xml ; then
	echo "Pass"
	true
else
	echo "Fail"
	false
fi
