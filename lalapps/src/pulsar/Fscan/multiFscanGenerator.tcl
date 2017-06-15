#!/bin/sh
# the next line starts using tclsh \
exec tclsh "$0" "$@"

proc SendMail { subject msg } {
  if {$::notifyList > ""} {
    if { [ exec uname ] == "SunOS" } {
      set mailer /usr/ucb/mail
      exec $mailer -s $subject $::notifyList << $msg
    } else {
      set mailer mail
      exec echo $msg | $mailer -s $subject $::notifyList
    }
  }
}

#procedure for printing to output file and archived copy
#proc Print {thisString}  {
#     puts $::outFID $thisString;
#     puts $::outArchiveFID $thisString;
#}

proc PrintHelpAndExit {} {
   global argv0;
   puts "";
   puts "Runs the fscan DAG generator, fscanDriver.py, for a tcl list of channels given in the resourceFile.";
   puts "";
   puts "Usage:";
   puts "";
   puts "$argv0 <resourceFile> \[segFile\] \[-R\]";
   puts "";
   puts "The resourceFile describes channels and times to generate fscans and must be given; see exampleFscanGenerator.rsc."
   puts "";
   puts "The segFile is optional, and if given this must be the complete path to a file with a list of segments."
   puts "The segFile can also be a list of segment files in the format \"IFO1:Path_to_IFO1_segFile IFO2:Path_to_IFO2_segFile ...\""
   puts "For example: \"H1:/home/fscans/H1segs.txt L1:/home/fscans/L1segs.txt V1:/home/fscans/V1segs.txt\""
   puts "The segFile can also be set in the resourceFile. If the segFile is not set anywhere, and useLSCsegFind is"
   puts "set to 1 in the resource file, then commands set in the resource file are used to find segments (e.g., "
   puts "using ligolw_segment_query and ligolw_print or find_locked_segments.py). The typeLSCsegFind is then a list of types to."
   puts "use for each IFO with ligolw_segment_query, or the list of IFO:TYPE:PathToIniFile's to use with find_locked_segments.py."
   puts "Otherwise the default is to generate fscans for ALL times between the start and end times set in the resourceFile."
   puts "";
   puts "If the -R option is given, then fscanDriver.py will be run with the -R option to submit the DAGs to condor."
   puts "";
   exit;
}

proc getColumn {Line ColNum} {
# Break a Line into columns and return what in column = ColNum.
	if {$ColNum < 1} {
		return "bad Column Number";
	}
	set LineList [split $Line];
	set ColumnList "";
	foreach ColumnItem $LineList {
		if {[string trimleft $ColumnItem] > ""} {
			lappend ColumnList $ColumnItem;
		}
	}
	return [lindex $ColumnList [expr $ColNum - 1]]; # Note first column has index 0.
}

proc getLSCsegFindsegs {typeLSCsegFind startTime endTime segFile} {
  if {[string match "*ligolw_segment_query*" $::segFindCmdAndPath]} {
   #set segCommand "exec $::segFindCmdAndPath --database --query-segments --include-segments $typeLSCsegFind --gps-start-time  $startTime --gps-end-time $endTime | $::grepCommandAndPath -v \"0, 0\" | $::ligolwPrintCommandAndPath -t segment:table -c start_time -c end_time -d \" \" > $segFile"
   #set segCommand "exec $segFindCmdAndPath --server $segFindServer --type $typeLSCsegFind --interferometer $thisIfoList2 --gps-start-time $startTime --gps-end-time $endTime";
   #set segCommand "exec $::segFindCmdAndPath --segment-url=https://dqsegdb5.phy.syr.edu --query-segments --include-segments $typeLSCsegFind --gps-start-time  $startTime --gps-end-time $endTime | $::grepCommandAndPath -v \"0, 0\" | $::ligolwPrintCommandAndPath -t segment:table -c start_time -c end_time -d \" \" > $segFile"
   set segCommand "exec $::segFindCmdAndPath --segment-url=https://segments.ligo.org --query-segments --include-segments $typeLSCsegFind --gps-start-time  $startTime --gps-end-time $endTime | $::grepCommandAndPath -v \"0, 0\" | $::ligolwPrintCommandAndPath -t segment:table -c start_time -c end_time -d \" \" > $segFile"
   puts "segCommand = $segCommand";
   if { [catch {eval $segCommand} tmpLSCsegFindsegs] } {
      puts "Error finding segments: $tmpLSCsegFindsegs"
      PrintHelpAndExit;
   } else {
      puts "Successfully output segments to: $segFile"
   }
  } elseif {[string match "*find_locked_segments.py*" $::segFindCmdAndPath]} {
   # In the .rsc file need a line like 
   # set typeLSCsegFind "H1:H1_M:psl_locked.ini" 
   # to set the IFO, type and ini file to use with find_locked_segments.py.
   set typeLSCsegFindItemList [split $typeLSCsegFind ":"];
   set thisIFO [lindex $typeLSCsegFindItemList 0];
   set thisType [lindex $typeLSCsegFindItemList 1];
   set thisConfig [lindex $typeLSCsegFindItemList 2];
   set segCommand "exec $::segFindCmdAndPath -s $startTime -e $endTime -i $thisIFO -t $thisType -c $thisConfig"
   puts "segCommand = $segCommand";
   if { [catch {eval $segCommand} tmpLSCsegFindsegs] } {
      puts "Error finding segments: $tmpLSCsegFindsegs"
      PrintHelpAndExit;
   }
   set duration [expr $endTime - $startTime];
   set tmpOutputSegFile "$thisIFO-LOCKED_SEGMENTS_"
   append tmpOutputSegFile $thisType;
   append tmpOutputSegFile "-$startTime-$duration";
   append tmpOutputSegFile ".txt";
   # set segCommand "exec /bin/cat $tmpOutputSegFile | /bin/grep -v \"#\" | /usr/bin/awk { '\{print \$2 \" \" \$3\}' } > $segFile"
   # puts "segCommand = $segCommand";
   # Instead read in the segments from columns 2 and 3 of the tmpOutputSegFile file.
   if { [catch {getSegFilesegs $tmpOutputSegFile 0 0 $startTime $endTime 2 3} tmpLSCsegFindsegs] } {
      puts "Error finding segments: $tmpLSCsegFindsegs"
      PrintHelpAndExit;
   } else {
      set segFID [open $segFile w];
      foreach seg $::segsFromFile {
        puts $segFID "$seg";
      }
      close $segFID;
      puts "Successfully output segments to: $segFile"
   }
  } else {
      puts "Error, did not recognize segFindCmdAndPath: $::segFindCmdAndPath"
      PrintHelpAndExit;
  }
}

proc getSegFilesegs {segFile incSegStartsBy decSegEndsBy startTime endTime segFileStartCol segFileEndCol} {
      # Open the segFile and read in the list of Segments.
      set listLines [list ];
      set segFID [open $segFile r];
      set segBuffer [read $segFID];
      close $segFID;
      # Split file into list of lines
      foreach line [split $segBuffer "\n"] {
          set line [string trimleft $line]; #trim leading spaces.
          if {[string index $line 0] == "#" || $line == ""} {
                    # line is blank or a comment, skip
          } else {
            lappend listLines $line;
          }
      }
      # Build the segment list      
      set ::segsFromFile {};
      foreach thisInterval $listLines {           
          set start_time [getColumn $thisInterval $segFileStartCol];
          set end_time [getColumn $thisInterval $segFileEndCol];
          # Increment/Decrement interval as specified by incSegStartsBy and decSegEndsBy.
          set start_time [expr $start_time + $incSegStartsBy];
          set end_time [expr $end_time - $decSegEndsBy];
          if {$end_time > $start_time} {
            # Check that interval overlaps with startTime endTime.
            if { ($start_time < $endTime) && ($end_time > $startTime) } {
              set addInterval [list $start_time $end_time];
              set ::segsFromFile [ setIntervalUnion $::segsFromFile $addInterval ];
            }
          }
      }
      #puts "::segsFromFile = $::segsFromFile"
}


proc moveSFTs {originalDir moveSFTsFromDirList moveSFTsFromSuffix moveSFTstoDir startTime endTime thisChanDir} {

  # mkdir the destination directory for the SFTs
  set cmd "exec mkdir -p $moveSFTstoDir";
  puts "Preparing to move SFTs; running $cmd";
  puts " ";
  eval $cmd

  # Loop through the parent dir location of SFTs, find the SFTs under this location for this channel, and move them to $moveSFTstoDir 
  foreach moveSFTsFromDir $moveSFTsFromDirList {

     set moveSFTCount 0;

     # Move the location of the SFTs
     cd $moveSFTsFromDir;

     # Get a list of directories that contain SFTs
     set fullPathDirList [glob -nocomplain "*/$thisChanDir/$moveSFTsFromSuffix"];
     set fullPathDirList [lsort $fullPathDirList];
     #puts $fullPathDirList;

     # Check if this location has SFTs between the start time and end time
     foreach dirPath $fullPathDirList {
        set pathList [split $dirPath "/"];
        set thisDateAndTime [lindex $pathList 0];
        #puts $thisDateAndTime
        set thisDateAndTimeList [split $thisDateAndTime "_"];
        set thisYear [lindex $thisDateAndTimeList 1];      # Get the yyyy year
        set thisMonthNum [lindex $thisDateAndTimeList 2];  # Get the numerical month (01, 02, 03, ....12)
        set thisMonthName $::monthNames($thisMonthNum);    # Get month name (Jan, Feb, Mar,...)
        set thisDayNum [lindex $thisDateAndTimeList 3];    # Get the numerical day (01, 02, 03, ....31)
        set thisTime [lrange $thisDateAndTimeList 4 6];    # Get time of day
        set thisHour [lindex $thisTime 0];                 # Get hour of the day
        if {$thisHour > 23} {
           # Handle bug where hour can be 24.
           set thisHour [expr $thisHour - 12];
           set thisMin [lindex $thisTime 1];
           set thisSec [lindex $thisTime 2];               # (format hh:mm:ss)
           set thisTime "$thisHour:$thisMin:$thisSec";
        } else {
           set thisTime [join $thisTime ":"];              # (format hh:mm:ss)
        }
        set thisTimeZone [lindex $thisDateAndTimeList 7];  # Get the time zone (PDT, CDT, ...)
        set thisGPSTime [tconvert "$thisMonthName $thisDayNum $thisYear $thisTime $thisTimeZone"]
        #puts "This date and time = $thisMonthName $thisDayNum $thisYear $thisTime $thisTimeZone";
        #puts "This GPS Time = $thisGPSTime";
        # Note using greater than $startTime and not greater than or equal $startTime
        # because directory give names give the end time of the last SFTs in that directory.
        if {($thisGPSTime > $startTime) && ($thisGPSTime <= $endTime)} {
           set sftList [glob -nocomplain "$dirPath/*.sft"];
           foreach sft $sftList {
              set cmd "exec mv $sft $moveSFTstoDir";
              #puts $cmd;
              eval $cmd
              incr moveSFTCount;
           }
        }
     }
     puts "Moved $moveSFTCount SFTs from $moveSFTsFromDir/*/$thisChanDir/$moveSFTsFromSuffix to $moveSFTstoDir";
     puts " ";
  }

  # Return to the original directory
  cd $originalDir
}

##### SET DEFAULT VALUES AND SOURCE FILES #####

#package require tconvert
source tconvert.tcl;

set startingPWD [exec pwd];

# Defaults; these should be set in the resource file

set ::masterList [list ]; # This is a tcl list with list of channels, frame types, etc....

set parentOutputDirectory "none";

# CURRENTLY: a file with a list of start_time end_time segments can be given, else all times are used.
set segFile "none"

# NONE OF THE OTHER SEGMENT METHODS ARE IMPLEMENTED:

set runDAGs 0;

set baseOutputSegmentFile "tmpFscanGenSegList.txt";

set useLSCsegFind 0;           # Use LSCsegFind to get segments.
set segFindCmdAndPath "";      # Path to LSCsegFind command needs to be set in .rsc file
set segFindServer     "";      # Server to use with LSCsegFind needs to be set in .rsc file
set typeLSCsegFind "Science";  # Comma deliminated string of types to use with LSCsegFind (e.g., Science)
set incSegStartsBy 0;          # Increase the start times of segs returned from LSCsegFind by this amount
set decSegEndsBy 0;            # Decrease the end times of segs returned from LSCsegFind by this amount
set ::LSCsegFindList {};       # 10/17/06 gam; initialize list of LSCsegFind segments;

set useSegFileList 0;          # If True, then use SegFileList(IFO) as segFile for each IFO.

set segFileStartCol   1;       # Column with segment start times in segment file
set segFileEndCol     2;       # Column with segment end times in segment file
set ::segsFromFile {};         # Initialize list of segFile segments;

set intersectData 0; # 0 == false, 1 == true; if true run fscanDriver.py with -I option so that it intersects the segments with the times data exists.

set fscanDriverPath "none"; # complete path to fscanDriver.py
set matlabPath "none";      # Path to matlab installation to use with -m option to fscanDriver.py, e.g., /ldcg/matlab_r2008a

set nGPSDigitsList [list 4]
set startTime "none"
set startTimeFile "none";
set endTime "none";

set useStartTimeForDirName 0;  # 0 == false, 1 == true; Use start time to name output directory.
set useEndTimeForDirName 0;    # 0 == false, 1 == true; Use start time to name output directory.


set timeLag "0";
set ::notifyList "";

set ::urlPrefix  "";

# Set up the numbers of the months
set monthNums(Jan) "01";set monthNums(Feb) "02";set monthNums(Mar) "03";set monthNums(Apr) "04";
set monthNums(May) "05";set monthNums(Jun) "06";set monthNums(Jul) "07";set monthNums(Aug) "08";
set monthNums(Sep) "09";set monthNums(Oct) "10";set monthNums(Nov) "11";set monthNums(Dec) "12";

# Set month names
set ::monthNames(01) "Jan";set ::monthNames(02) "Feb";set ::monthNames(03) "Mar";
set ::monthNames(04) "Apr";set ::monthNames(05) "May";set ::monthNames(06) "Jun";
set ::monthNames(07) "Jul";set ::monthNames(08) "Aug";set ::monthNames(09) "Sep";
set ::monthNames(10) "Oct";set ::monthNames(11) "Nov";set ::monthNames(12) "Dec";

# If fixedComparison is 1 then used then ignore the comparison delta dir in the masterList but compare using fixd values:
set fixedComparison 0;
set fixedComparisonChanDir "";
set fixedComparisonString "";
set fixedComparisonSNR "";

# If these are set then SFTs are moved from the moveSFTsFromDirList rather than generated. 
set moveSFTsFromDirList "";
set moveSFTsFromSuffix "";

set crab 0; #if crab stuff is to be run then rsc file sets crab to 1, default is not to run.X

set submitLogFileDir "/usr1/pulsar"; # directory location for log files given in the condor submit files, for use with fscanDriver.py -o option.

##### MAIN CODE STARTS HERE #####

# Parse command line arguements

if {$argv == "" || $argv == "-h" || $argv == "?" || $argv == "--help"} {
   PrintHelpAndExit;
}

if {$argc < 1} {
   PrintHelpAndExit;
}

# source the resource file
set resourceFile [lindex $argv 0];
source $resourceFile

source sets.tcl; # source the procedures for handling segment lists

if {$parentOutputDirectory == "none"} {
   puts "Error: parentOutputDirectory not set in $resourceFile";
   PrintHelpAndExit;
}

set startTimeFromFile "";
if {$startTimeFile == "none"} {
  # continue
} else {
  if { [catch {exec cat $startingPWD/$startTimeFile} startTimeFromFile] } {
   # continue; the startTime could be set in the resource file.
  } else {
   set startTime $startTimeFromFile;
  }
}

if {$startTime == "none"} {
   puts "Could not determine start time from $resourceFile or $startTimeFromFile.";
   PrintHelpAndExit;
}

if {$endTime == "none"} {
   puts "Could not determine end time from $resourceFile.";
   PrintHelpAndExit;
} elseif {$endTime == "now"} {
  set endTime [ tconvert now ];
  set endTime [ expr $endTime - $timeLag];
} elseif { [string index $endTime 0] == "+"} {
  set endTime [string range $endTime 1 end];
  set endTime [ expr $startTime + $endTime];
}

# Update the startTimeFile before any errors occur to prevent fscans from running again on this time.
if {$startTimeFile == "none"} {
  # continue
} else {
  set cmd "exec echo $endTime > $startingPWD/$startTimeFile";
  eval $cmd;
}

set duration [expr $endTime - $startTime];

if {$fscanDriverPath == "none"} {
   puts "Could not determine path to fscanDriver.py from $resourceFile.";
   PrintHelpAndExit;
}

if {$matlabPath == "none"} {
   puts "Could not determine matlab path from $resourceFile.";
   PrintHelpAndExit;

}

# Format start and end time
set formatStartTime [tconvert -l -f "%a %b %d %I:%M:%S %p %Z %Y" $startTime]
set formatEndTime [tconvert -l -f "%a %b %d %I:%M:%S %p %Z %Y" $endTime]

# Get the date and time from start time, end time or now, depending flags set, and reformat for easy sorting by year, month, numberical day, etc...
if {$useStartTimeForDirName} {
   set formatTimeNow $formatStartTime;
} elseif {$useEndTimeForDirName} {
   set formatTimeNow $formatEndTime;
} else {
   set formatTimeNow [clock format [clock seconds]];
}
set timeYear [lindex $formatTimeNow 6];
set timeMonthChar [lindex $formatTimeNow 1];
set timeMonthNum $monthNums($timeMonthChar);
set timeDayNum [lindex $formatTimeNow 2];
set timeDayChar [lindex $formatTimeNow 0];
set timeAMPM [lindex $formatTimeNow 4];
set timeTime [split [lindex $formatTimeNow 3] ":"]
# Convert to 24 hr time
if {$timeAMPM == "PM"} {
   set timeTimeHr [string trimleft [lindex $timeTime 0] "0"];
   if {$timeTimeHr < 12} {
      set timeTimeHr [expr $timeTimeHr + 12]; # Only add 12 for 1 to 11 PM, not for 12 PM.
   }
   set timeTime [list $timeTimeHr [lindex $timeTime 1] [lindex $timeTime 2]];
}
set timeTime [join $timeTime "_"];
set timeZone [lindex $formatTimeNow 5];
set joinFormatTimeNow [join [list $timeYear $timeMonthNum $timeDayNum $timeTime $timeZone $timeDayChar] "_"]

set channelList [list ];
set chanDirList [list ];
set chanTypeList [list ];
set ifoList [list ];
set frameTypeList [list ];
set sftOutputDirList [list ];
set comparisonChanList [list ];
set comparisonSNRList [list ];
set comparisonDeltaDirList [list ];
set kneeFreqList [list ];
set sftTimeBaseList [list ];
set startFreqList [list ];
set bandList [list ];
set subBandList [list ];
set extraTimeList [list ];
set freqResList [list ];
set chanAltDirName "";
if {$crab != 0} {
    set psrInputList [list ];
    set psrEphemerisList [list ];
    set earthFileList [list ];
    set sunFileList [list ];
}

#for {set i 0} {$i < [llength $::masterList]} {incr i} {}
foreach item $::masterList {
    puts $item
    set thisChan [lindex $item 0]
    lappend channelList $thisChan
    set chanAltDirName [lindex $item 15];
    # If an alternative name for the channel directory is not == default, use this rather than the channel name itself as the directory for this channel.
    if {$chanAltDirName == "default"} {
      lappend chanDirList [join [split $thisChan ":"] "_"]
    } else {
      lappend chanDirList $chanAltDirName
    }
    lappend chanTypeList [lindex $item 1]
    lappend ifoList [lindex $item 2]
    lappend frameTypeList [lindex $item 3]
    lappend sftOutputDirList [lindex $item 4]
    lappend comparisonChanList [lindex $item 5]
    lappend comparisonSNRList [lindex $item 6]
    lappend comparisonDeltaDirList [lindex $item 7]
    lappend kneeFreqList [lindex $item 8];
    lappend sftTimeBaseList [lindex $item 9];
    lappend startFreqList [lindex $item 10];
    lappend bandList [lindex $item 11];
    lappend subBandList [lindex $item 12];
    lappend extraTimeList [lindex $item 13];
    lappend freqResList [lindex $item 14];
    if {$crab != 0} {
        lappend psrInputList [lindex $item 16];
        lappend psrEphemerisList [lindex $item 17];
        lappend earthFileList [lindex $item 18];
        lappend sunFileList [lindex $item 19];
    }
}

#puts "";
#puts $channelList; puts "";
#puts $chanDirList; puts "";
#puts $chanTypeList; puts "";
#puts $ifoList; puts "";
#puts $frameTypeList; puts "";
#puts $sftOutputDirList; puts "";
#puts $comparisonChanList; puts "";
#puts $comparisonSNRList; puts "";
#puts $comparisonDeltaDirList; puts "";

# segFile can be set in the resource file, but if given as second argument, then this is used:
if {$argc >= 2} {
   if {[lindex $argv 1] != "-R"} {
     set segFile [lindex $argv 1];
   }
}

if {$segFile != "none"} {
  # $segFile needs to be complete path to the file
  if {[llength $segFile] > 1} {
     # If list given then format is IFO1:path_to_IFO1_segments IFO2:path_toIFO2_segments ... 
     set useSegFileList 1;
     foreach segFileItem $segFile {
             #puts "segFileItem = $segFileItem";
             set segFileItemList [split $segFileItem ":"];
             set thisIFO [lindex $segFileItemList 0];
             set thisSegFilePath [lindex $segFileItemList 1];
             puts "thisSegFilePath = $thisSegFilePath for thisIFO = $thisIFO";
             set segFileList($thisIFO) $thisSegFilePath;
     }
  } else {
     # continue;
  }
} else {
  if {$useLSCsegFind} {
    if {[llength $typeLSCsegFind] > 1} {
      set useSegFileList 1;
      foreach typeLSCsegFindItem $typeLSCsegFind {
        set typeLSCsegFindItemList [split $typeLSCsegFindItem ":"];
        set thisIFO [lindex $typeLSCsegFindItemList 0];
        set thisSegFilePath "$startingPWD/multiFscanSegFile_$thisIFO";
        append thisSegFilePath "_";
        append thisSegFilePath $joinFormatTimeNow.txt;
        set segFileList($thisIFO) $thisSegFilePath;
        puts "thisSegFilePath = $thisSegFilePath for thisIFO = $thisIFO";
        set ::LSCsegFindList {}; # initialize list of LSCsegFind segments;
        getLSCsegFindsegs $typeLSCsegFindItem $startTime $endTime $thisSegFilePath;
      }
    } else {
      # Do as before
      set segFile "$startingPWD/multiFscanSegFile_$joinFormatTimeNow.txt"
      puts "segFile = $segFile"
      set ::LSCsegFindList {}; # initialize list of LSCsegFind segments;
      getLSCsegFindsegs $typeLSCsegFind $startTime $endTime $segFile;
    }
  } else {
    set segFile "ALL"; # do all times
  }
}

# Check whether to run the DAGs.
if {$argc >= 3} {
   if {[lindex $argv 2] == "-R"} {
      set runDAGs 1;
   }
} elseif {$argc >= 2} {
   if {[lindex $argv 1] == "-R"} {
      set runDAGs 1;
   }
}

# Change to the directory where the code should run
cd $parentOutputDirectory;

# Get list of existing directories (inludes path to html files to parse out GPS times for each fscan dir).
set previousDirList [glob -nocomplain "*"];
set previousDirList [lsort $previousDirList];
set previousDirListLength [llength $previousDirList];

set thisOutputDir "fscans_$joinFormatTimeNow";
set cmd "exec mkdir $thisOutputDir";
eval $cmd
cd $thisOutputDir
puts $cmd; puts "";

for {set i 0} {$i < [llength $::masterList]} {incr i} {

    set thisChan [lindex $channelList $i];
    set thisChanDir [lindex $chanDirList $i];
    set thisChanType [lindex $chanTypeList $i];
    set thisIFO  [lindex $ifoList $i];
    set thisFrameType [lindex $frameTypeList $i];
    set thisSFTOutDir [lindex $sftOutputDirList $i];
    set thisComparisonChan [lindex $comparisonChanList $i];
    set thisComparisonSNR [lindex $comparisonSNRList $i];
    set thisComparisonDeltaDir [lindex $comparisonDeltaDirList $i];
    set thisKneeFreq [lindex  $kneeFreqList $i];
    set thisSFTTimeBase [lindex $sftTimeBaseList $i];
    set thisStartFreq [lindex  $startFreqList $i];
    set thisBand [lindex $bandList $i];
    set thisSubBand [lindex $subBandList $i];
    set thisExtraTime [lindex $extraTimeList $i];
    set thisFreqRes [lindex $freqResList $i];
    if {$crab != 0} {
        set thispsrInput [lindex $psrInputList $i];
        set thispsrEphemeris [lindex $psrEphemerisList $i];
        set thisearthFile [lindex $earthFileList $i];
        set thissunFile [lindex $sunFileList $i];
    }

    set cmd "exec mkdir $thisChanDir";
    eval $cmd

    # Change to the current directory and run the fscan DAG generator
    cd $thisChanDir;
    set thisDir [exec pwd];
    #set thisG "$thisChanDir_$thisIFO_$startTime_$endTime";
    set thisG "$thisChanDir";
    append thisG "_";
    append thisG "$thisIFO";
    append thisG "_";
    append thisG "$startTime";
    append thisG "_";
    append thisG "$endTime";
    if {$thisSFTOutDir == "default"} {
       set thisSFTOutDir $thisDir/sfts 
    }
    set cmd "exec $fscanDriverPath";
    # Check whether we are moving existing SFTs for creating SFTs
    if {($moveSFTsFromDirList > "") && ($moveSFTsFromSuffix > "")} {
       # Do not give -C option to create SFTs, instead move SFTs to run on.
       moveSFTs $thisDir $moveSFTsFromDirList $moveSFTsFromSuffix $thisSFTOutDir $startTime $endTime $thisChanDir;
       append cmd " -s $startTime -L $duration -G $thisG -d $thisFrameType -k $thisKneeFreq -T $thisSFTTimeBase -u $thisChanType -i $thisIFO";
    } else {
       append cmd " -s $startTime -L $duration -G $thisG -d $thisFrameType -k $thisKneeFreq -T $thisSFTTimeBase -C -u $thisChanType -i $thisIFO";
    }
    append cmd " -p $thisSFTOutDir -N $thisChan -F $thisStartFreq -B $thisBand -b $thisSubBand";
    append cmd " -X fscan$thisChanDir -o $submitLogFileDir"
    append cmd " -O $thisDir -m $matlabPath"
    append cmd " -W $thisDir/fscan$thisG.html"
    if {$useSegFileList} {
       # Use the segFile that goes with thisIFO.
       set segFile $segFileList($thisIFO);
    }
    append cmd " -x $thisExtraTime -v 2 -g $segFile --freq-res $thisFreqRes"
    # Set up comparison with another fscan:
    if {$crab != 0} {
       append cmd " -J $thispsrInput -j $thispsrEphemeris -E $thisearthFile -z $thissunFile";
    }
    if {$thisComparisonChan != "none"} {
       set thisComparisonChan [split $thisComparisonChan ":"]
       set thisComparisonIFO [lindex $thisComparisonChan 0]
       set thisComparisonChan [join $thisComparisonChan "_"]
       if {$fixedComparison} {
         set thisComparisonChanDir $fixedComparisonChanDir;
         set thisComparisonString $fixedComparisonString;
         set thisComparisonSNR $fixedComparisonSNR;
       } else {
         if {$thisComparisonDeltaDir < 0} {
           # Look in the list of paths ot previous fscans to get the path to the thisComparisonDeltaDir previous fscan:
           set thisComparisonDirPath [lindex $previousDirList [expr $previousDirListLength + $thisComparisonDeltaDir]];
           set thisComparisonChanDir "../../$thisComparisonDirPath/$thisComparisonChan";
           # Now find the path to previous html file to parse to get GPS times of that fscan:
           set thisComparisonDirPath [glob "$thisComparisonChanDir/fscan*.html"];
           set thisComparisonDirPath [split $thisComparisonDirPath "/"];
           # Get the comparison string for that fscan from the name of the html file for that fscan:
           set thisComparisonHTML [lindex $thisComparisonDirPath [expr [llength $thisComparisonDirPath] - 1]];
           set thisComparisonHTML [split $thisComparisonHTML "_"];
           set thisComparisonHTMLlength [llength $thisComparisonHTML];
           set thisComparisonStartTime [lindex $thisComparisonHTML [expr $thisComparisonHTMLlength - 2]];
           set thisComparisonEndTime [lindex $thisComparisonHTML [expr $thisComparisonHTMLlength - 1]];
           set thisComparisonEndTime [lindex [split $thisComparisonEndTime "."] 0];
           set thisComparisonString $thisComparisonIFO;
           append thisComparisonString "_";
           append thisComparisonString "$thisComparisonStartTime";
           append thisComparisonString "_";
           append thisComparisonString "$thisComparisonEndTime";
         } else {
           set thisComparisonChanDir "../$thisComparisonChan";
           set thisComparisonString $thisComparisonIFO;
           append thisComparisonString "_";
           append thisComparisonString "$startTime";
           append thisComparisonString "_";
           append thisComparisonString "$endTime";
         }
       }
       append cmd " -r $thisComparisonChanDir -e $thisComparisonString -q $thisComparisonSNR"
    }
    if {$intersectData} {
        # Append option to run ligo_data_find and intersect segments with times data exists.
        append cmd " -I";
    }
    if {$runDAGs} {
        # Append option to run the dags
        append cmd " -R";
    }
    puts "Running:\n$cmd"; puts "";
    if { [catch {eval $cmd} fscanDriverOut] } {
       puts "Message: $fscanDriverOut";
    }
    cd ..
}

cd $startingPWD;

#close $::outFID;
#close $::outArchiveFID;

#SendMail $subject $msg;

exit;
#### MAIN CODE ENDS HERE #####
