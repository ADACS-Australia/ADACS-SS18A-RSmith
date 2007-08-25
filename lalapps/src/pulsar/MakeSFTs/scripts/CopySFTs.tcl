#!/usr/bin/env tclshexe

# File: CopySFTs.tclsh Started: 08/24/07
# Author: Greg Mendell

# REVISIONS

# For dd and mv commands:
set ::DDLOCATION "/bin/dd"
set ::DDIBS {256k}
set ::DDOBS {256k}
set ::MVLOCATION "/bin/mv"
#set ::FINDORGLOB "exec /usr/bin/find"
set ::FINDORGLOB "glob"

proc CopyFiles {inputFileAndPath outputPath outputFileAndPath fidListInFiles fidListOutFiles fidListBadFiles path_to_SFTvalidate do_SFTvalidate path_to_md5sum do_md5sums VERBOSE} {

    if {$VERBOSE} {
       set clockIn [clock seconds];
    }

    # Validate the input file
    if {$do_SFTvalidate} {
       if {[catch {exec $path_to_SFTvalidate $inputFileAndPath} result]} {
          puts stderr "$inputFileAndPath failed $path_to_SFTvalidate; $result."
          puts $fidListBadFiles $inputFileAndPath;
          return;
       } else {
         if {$VERBOSE} {
            puts stdout "Input file passed $path_to_SFTvalidate."
         }
       }
    }

    # Get md5sum of input file
    if {$do_md5sums} {
       if {[catch {exec $path_to_md5sum $inputFileAndPath} inputMD5SUM]} {
          puts stderr "$inputFileAndPath failed $path_to_md5sum; $inputMD5SUM."
          puts $fidListBadFiles $inputFileAndPath;
          return;
       }
    }

    # Add to list of input files
    if {$fidListInFiles > -1} {
       puts $fidListInFiles $inputFileAndPath;
    }
    
    # Make sure output file does not already exist
    if {[file exists $outputFileAndPath]} {
       puts stderr "NOT copying $inputFileAndPath to $outputFileAndPath; $outputFileAndPath already exists.";
       return
    }

    # Make temporary name for output    
    set outputFileAndPathTmp "$outputFileAndPath.tmp";

    # Make output directory if necessary
    if {[file isdirectory $outputPath]} {
       # Continue; output path already exists. 
    } else {
       if {[catch {exec mkdir -p $outputPath} result]} {
          puts stderr "Could not make directory $outputPath; $result.";
          return;
       }
    }

    # Copy the file
    set cmd "exec $::DDLOCATION ibs=$::DDIBS obs=$::DDOBS if=$inputFileAndPath of=$outputFileAndPathTmp 2>@stdout";
    if {$VERBOSE} {
       puts stdout "Starting $cmd.";
    }
    if {[catch {eval $cmd} result]} {
       puts stderr "$cmd failed; $result";
       return;
    } else {
       # Validate the output file
       if {$do_SFTvalidate} {
          if {[catch {exec $path_to_SFTvalidate $outputFileAndPathTmp} result]} {
             puts stderr "$outputFileAndPathTmp failed $path_to_SFTvalidate; $result.";
             return;
          } else {
            if {$VERBOSE} {
               puts stdout "Output file passed $path_to_SFTvalidate."
            }
          }
       }
       
       # Check md5sum of output against input
       if {$do_md5sums} {
          if {[catch {exec $path_to_md5sum $outputFileAndPathTmp} outputMD5SUM]} {
             puts stderr "$outputFileAndPathTmp failed $path_to_md5sum; $outputMD5SUM."
             return;
          } else {
            if {[lindex [split $inputMD5SUM] 0] != [lindex [split $outputMD5SUM] 0]} {
               puts stderr "$path_to_md5sum comparison of $inputMD5SUM and $outputMD5SUM failed."
               return;
            } else {
              if {$VERBOSE} {
                 puts stdout "Input and output $path_to_md5sum agree."
              }
            }
          }
       }      
       
       # Move the tmp file:
       set mvcmd "exec $::MVLOCATION $outputFileAndPathTmp $outputFileAndPath";
       if {[catch {eval $mvcmd} result]} {
          puts stderr "$mvcmd failed; $result."
          return;
       }

    }
    # End if {[catch {eval $cmd} result]}

    # Write to list of output files
    if {$fidListOutFiles > -1} {
       puts $fidListOutFiles $outputFileAndPath;
    }
    
    if {$VERBOSE} {
       set clockOut [clock seconds];
       set clockTime [expr $clockOut - $clockIn];
       puts stdout "Copied $inputFileAndPath to $outputFileAndPath in $clockTime seconds.";
    }

}

proc Exit {code} {
     global argv0;
     if {$code > 2} {
           puts "The $argv0 script finished with an error."
        } elseif {$code == 1} {
           # Displayed help.  Just quit without a message.
        } elseif {$code == 2} {
           puts "Script terminated.";
        } else {
           puts "The $argv0 script has finished.";
        }
        exit $code;
}

proc PrintHelp {} {
    global argv0;
    puts " ";
    puts "Copy SFTs from paths on nodes to another filesystem or vice versa, with options to run SFTvalidate and check md5sums.";
    puts " ";    
    puts "Usage:";
    puts " ";
    puts "$argv0 -i <input_path> -o <output_path> -node_list <node_list_file> -node_flag <i|o> -n <num_per_node> -opan <output_path_after_node> -p <sft_pattern> -SFTval <path_to_SFTvalidate> -md5sums <path_to_md5sum> -list_in_files <filename_listInFiles> -list_out_files <filename_listOutFiles> -V"; 
    puts " ";
    puts "-V";        
    puts "-i <input_path>                         Path to input files.";
    puts "-o <output_path>                        Path to output files.";
    puts "-node_list <node_list_file>             File with list of nodes. (This is optional, but must include node_flag if given.)";
    puts "-node_flag <i|o>                        Flag: if equal to \"i\" then append to input_path; if \"o\" append to output_path.";
    puts "-n <num_per_node>                       If node_flag == o this is the number of files to put on each node in the node_list.";
    puts "-opan <output_path_after_node>          If node_flag == o this is the path to append after the node name in the output path.";
    puts "-p <sft_pattern>                        Pattern to append to input path (after node name if -node_flag == \"i\" to find SFT files.";
    puts " ";
    puts "Note that the directory above the SFTs is also copied.  For example, if input_path = /data and"
    puts "output_path = /archive/sfts, node_list = \[node1\], node_flag = i and sft_pattern =  sfts/LHO/*/*.sft"
    puts "then /data/node1/sfts/LHO/H-1_H1_1800SFT_C03hoft-815/H-1_H1_1800SFT_C03hoft-815827752-1800.sft";
    puts "gets copied to /archive/sfts/H-1_H1_1800SFT_C03hoft-815/H-1_H1_1800SFT_C03hoft-815827752-1800.sft."; 
    puts "Thus note that the directory name, H-1_H1_1800SFT_C03hoft-815, was also copied.";
    puts " ";        
    puts "-SFTval <path_to_SFTvalidate>           Path to SFTvalidate; if given then SFTvalidate is run on input and output files.";
    puts "-md5sums <path_to_md5sum>               Path to md5sum; if given then md5sums are run on input and output files.";
    puts "-list_in_files <filename_listInFiles>   If given, write list of files found under input_path into this file.";
    puts "-list_out_files <filename_listOutFiles> If given, write list of files copied to output_path into this file.";
    puts "-V                                      Write verbose output to stdout.";
    puts " ";    
    puts "A list of bad files found is written to badSFTsFoundByCopySFTs.<timestamp>.txt";
    puts " ";    
    puts " ";
    Exit 1;
}

#######################
# MAIN CODE STARTS HERE 
#######################

if { $argc < 1 || $argv == "-h" || $argv == "--help"} {
   PrintHelp;
}

# get the command line arguments
#puts "$argv0 -i <input_path> -o <output_path> -node_list <node_list_file> -node_flag <i|o> -n <num_per_node> -p <sft_pattern> -SFTval <path_to_SFTvalidate> -md5sums <path_to_md5sum> -list_in_files <filename_listInFiles> -list_out_files <filename_listOutFiles> -V";
set input_path "";
set output_path "";
set node_list_file "";
set node_flag "";
set num_per_node 0;
set output_path_after_node "";
set sft_pattern "";
set path_to_SFTvalidate "";
set do_SFTvalidate 0;
set path_to_md5sum "";
set do_md5sums 0;
set filename_listInFiles "";
set listInFiles 0;
set filename_listOutFiles "";
set listOutFiles 0;
set VERBOSE 0;
set optList [split "-i:-o:-node_list:-node_flag:-n:-opan:-p:-SFTval:-md5sums:-list_in_files:-list_out_files:-V" ":"];
set opt "unknown";
foreach element $argv {
        set optInd [lsearch -exact $optList $element];
        if {[lsearch -exact $optList $element] > -1} {
                set opt [lindex $optList $optInd];
                if {$opt == "-V"} {
                  set VERBOSE 1;
                }
        } else {
                if {$opt == "-i"} {
                        set input_path $element;
                } elseif {$opt == "-o"} {
                        set output_path $element;
                } elseif {$opt == "-node_list"} {
                        set node_list_file $element;
                } elseif {$opt == "-node_flag"} {
                        set node_flag $element;
                } elseif {$opt == "-n"} {
                        set num_per_node $element;
                } elseif {$opt == "-opan"} {
                        set output_path_after_node $element;
                } elseif {$opt == "-p"} {
                        set sft_pattern $element;
                } elseif {$opt == "-SFTval"} {
                        set path_to_SFTvalidate $element;
                        set do_SFTvalidate 1;
                } elseif {$opt == "-md5sums"} {
                        set path_to_md5sum $element;
                        set do_md5sums 1;
                } elseif {$opt == "-list_in_files"} {
                        set filename_listInFiles $element;
                        set listInFiles 1;
                } elseif {$opt == "-list_out_files"} {
                        set filename_listOutFiles $element;
                        set listOutFiles 1;
                }
                set opt "unknown";
        }
}

# Validate command line args and initial set up:
if {$input_path == ""} {
  puts "Error: no input_path";
  Exit 2;
}
if {$output_path == ""} {
  puts "Error: no output_path";
  Exit 2;
}
set nodeList [list ];
if {$node_list_file > ""} {
   if {($node_flag != "i") && ($node_flag != "o") } {
      puts "Error: node_flag must be i or o.";
      Exit 2;
   }
   set appendNodes 1;
   if {[file exists $node_list_file]} {
       # Read the file into buffer.
       set fid [open $node_list_file "r"];
       set buffer [read $fid];
       close $fid;
     } else {
       puts "Error: node_list_file = $node_list_file does not exist.";
       Exit 2;
     }
     foreach line [split $buffer "\n"] {
          set line [string trimleft $line]; #trim leading spaces.
          if {[string index $line 0] == "#" || $line == ""} {
                    # line is blank or a comment, skip
          } else {
            lappend nodeList $line;
          }
     }

} else {
   set appendNodes 0;
}
if {($node_flag == "o") && ($num_per_node < 1) } {
  puts "Error: if outputing files to nodes, then num_per_node must be >= 1.";
  Exit 2;
}
if {$sft_pattern == ""} {
  puts "Error: no sft_pattern";
  Exit 2;
}
if {$filename_listInFiles > ""} {
   set fidListInFiles [open $filename_listInFiles "w"];
} else {
   set fidListInFiles -1;
}
if {$filename_listOutFiles > ""} {
   set fidListOutFiles [open $filename_listOutFiles "w"];
} else {
   set fidListOutFiles -1;
}

# Make file with list of bad input files:
set formatTimeNow [clock format [clock seconds]];
set joinFormatTimeNow [join [join [split $formatTimeNow ":"] "_"] "_"]; # Need to join twice to join all with "_"
set listBadFiles "badSFTsFoundByCopySFTs.$joinFormatTimeNow.txt";
set fidListBadFiles [open $listBadFiles "w"];
  
# Go through input paths and get list of SFTs, validate if requested and copy these to output paths:
if {$node_flag == "i"} {
   # Include node names if given in node_list in the input paths.
   if {$appendNodes} {
      set count [llength $nodeList];
   } else {
     set count 1;
   }
   for {set j 0} {$j < $count} {incr j} {
       set inputPath $input_path;
       if {$appendNodes} {
          append inputPath "/";
          append inputPath [lindex $nodeList $j];
       }
       append inputPath "/";
       append inputPath $sft_pattern;
       #if {[catch {set sft_list [glob $inputPath]} result]}
       set findOrGlobCmd "$::FINDORGLOB $inputPath"
       if {[catch {set sft_list_out [eval $findOrGlobCmd]} result]} {
          puts stderr "$findOrGlobCmd failed; $result";
       } else {
         set sft_list [lsort $sft_list_out];
         foreach inputFileAndPath $sft_list {
               set outputFileAndPath $output_path;
               append outputFileAndPath "/";
               set splitFileAndPath [split $inputFileAndPath "/"];
               set lengthSplitFileAndPath [llength $splitFileAndPath];
               set lengthSplitFileAndPathm2 [expr $lengthSplitFileAndPath - 2];
               set sftFileName [lindex $splitFileAndPath end];
               if {$lengthSplitFileAndPathm2 > -1} {
                  set sftDirName [lindex $splitFileAndPath $lengthSplitFileAndPathm2];
                  append outputFileAndPath $sftDirName;
                  append outputFileAndPath "/";
               }
               set outputPath $outputFileAndPath;
               append outputFileAndPath $sftFileName;
               CopyFiles $inputFileAndPath $outputPath $outputFileAndPath $fidListInFiles $fidListOutFiles $fidListBadFiles $path_to_SFTvalidate $do_SFTvalidate $path_to_md5sum $do_md5sums $VERBOSE;
         }
         # END foreach inputFileAndPath $sft_list
       }
       # END if {[catch {set sft_list [glob $inputPath]} result]} else
   }
   # END    for {set j 0} {$j < $count} {incr j}
} elseif {$node_flag == "o"} {
  # Include node names if given in node_list in the output paths.
  set inputPath $input_path;
  append inputPath "/";
  append inputPath $sft_pattern;
  #if {[catch {set sft_list [glob $inputPath]} result]}
  set findOrGlobCmd "$::FINDORGLOB $inputPath"
  if {[catch {set sft_list_out [eval $findOrGlobCmd]} result]} {
     puts stderr "$findOrGlobCmd failed; $result";
  } else {
    set sft_list [lsort $sft_list_out];
    set numCountOnNode 0;
    set nodeIndex 0;
    foreach inputFileAndPath $sft_list {
        set outputFileAndPath $output_path;
        if {$appendNodes} {
           # increment the number of output files on this node
           incr numCountOnNode;
           if {$numCountOnNode > $num_per_node} {
              incr nodeIndex;       # Start output on the next node
              set numCountOnNode 1; # reset the count to 1; this is the 1st file on the next node.
           }
           append outputFileAndPath "/";
           append outputFileAndPath [lindex $nodeList $nodeIndex];
           # Add in additional path on the node
           if {$output_path_after_node > ""} {
              append outputFileAndPath "/";
              append outputFileAndPath $output_path_after_node;
           }
        }
        append outputFileAndPath "/";
        set splitFileAndPath [split $inputFileAndPath "/"];
        set lengthSplitFileAndPath [llength $splitFileAndPath];
        set lengthSplitFileAndPathm2 [expr $lengthSplitFileAndPath - 2];
        set sftFileName [lindex $splitFileAndPath end];
        if {$lengthSplitFileAndPathm2 > -1} {
            set sftDirName [lindex $splitFileAndPath $lengthSplitFileAndPathm2];
            append outputFileAndPath $sftDirName;
            append outputFileAndPath "/";
        }
        set outputPath $outputFileAndPath;
        append outputFileAndPath $sftFileName;
        CopyFiles $inputFileAndPath $outputPath $outputFileAndPath $fidListInFiles $fidListOutFiles $fidListBadFiles $path_to_SFTvalidate $do_SFTvalidate $path_to_md5sum $do_md5sums $VERBOSE;
    }
    # END foreach inputFileAndPath $sft_list
  }
  # END if {[catch {set sft_list [glob $inputPath]} result]} else
}
# ENd if {$node_flag == "i"} else

if {$filename_listInFiles > ""} {
   close $fidListInFiles;
}
if {$filename_listOutFiles > ""} {
   close $fidListOutFiles;
}

close $fidListBadFiles;

Exit 0;
