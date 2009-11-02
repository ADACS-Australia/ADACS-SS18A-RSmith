

# Configuration Script for the coherent followup pipeline.

foreach {var value} {
	search_name  "pf_S5_outliers1"
	work_dir  "/archive/home/josephb/PowerFluxFollowUpResamp/$search_name"
	
	fstat_binary  "/archive/home/josephb/PowerFluxFollowUpResamp/lalapps_ComputeFStatistic_resamp_orig"

	parameter_source_file  "/archive/home/josephb/PowerFluxFollowUpResamp/test_outliers1.csv"

	ra_band  0.05
	dec_band  0.05
	
	f0_band  1e-2
	f1_band  5e-11
	f2_band  0
	f3_band  0
	
	f2 0
	f3 0
	
	coherence_time  "[expr 14*24*60*60]"
	param_time  846885755
	start_time  815155213
	end_time  877824097
	individual_IFOs  False
	iteration  0
	coherence_time_multiplier_per_iteration  4.0
	parameter_space_multiplier  10
	
	false_alarm_cutoff  0.10
	
	num_candidates_to_keep  1000
	min_Fstat_to_keep  0
	percent_largest_to_average  0.90
	
	grid_type  2
	metric_type  1
	mismatch  0.15
	
	max_number_of_jobs  1
	SFT_length  1800
	
	frequency_band_wings  2.0
	psd_estimation_wings 0.2
	
	soft_link_sfts_only  True
	
	messy  True
	
	ephem_dir  "/archive/home/josephb/head/opt/lscsoft/lal/share/lal"
	ephem_year  "05-09"
	
	sft_location_files  {
		H1 "/archive/home/josephb/followScript/TotalListOnNodes_H1" 
		L1  "/archive/home/josephb/followScript/TotalListOnNodes_L1"
		}
	
	log_file_dir "/usr1/josephb"
	false_alarm_results_directory "$work_dir/false_alarm"
	events_directory "$work_dir/events"
	script_dir  "/archive/home/josephb/PowerFluxFollowUpResamp"
	psd_dir "/archive/home/josephb/freq_band_psd"

	main_dag_file_name "$work_dir/${search_name}.dag"
	main_condorA_sub "$work_dir/${search_name}_prepare.sub"
	main_condorB_sub "$work_dir/${search_name}_submit.sub"
	
	veto_segment_file_name unknown
	segment_file_name unknown
	
	
	} {
	global $var
	set $var [subst $value]
	#puts "$var [set $var]"
	}


set main_condorA_sub_text {
universe= vanilla 
executable = ${script_dir}/FstatFollowUp.tcl
output = $work_dir/std_out_err/node_${search_name}_A.out.\$(JobID)
error = $work_dir/std_out_err/node_${search_name}_A.err.\$(JobID)
log = ${log_file_dir}/${search_name}.log
notify_user=jcbetzwieser@gmail.com
arguments = \$(arglist)
requirements = Memory > 500
should_transfer_files = NO
queue
}

set main_condorB_sub_text {
universe= scheduler
getenv= true
executable = ${script_dir}/MainSubmit.tcl
output = $work_dir/std_out_err/node_${search_name}_B.out.\$(JobID)
error = $work_dir/std_out_err/node_${search_name}_B.err.\$(JobID)
log = ${log_file_dir}/${search_name}.log
notify_user=jcbetzwieser@gmail.com
arguments = \$(arglist)
requirements = Memory > 500
should_transfer_files = NO
queue
}