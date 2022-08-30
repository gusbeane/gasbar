#*******************************************************************************
# This file is part of the GALIC code developed by D. Yurin and V. Springel.
#
# Copyright (c) 2014
# Denis Yurin (denis.yurin@h-its.org) 
# Volker Springel (volker.springel@h-its.org)
#*******************************************************************************
#
# This file processes the configurations options in Config.sh, producing 
# two files:
#
#   galicconfig.h          to be included in each source file (via allvars.h)
#   compile_time_info.c    code to be compiled in, which will print the configuration 
#
if( @ARGV != 2)
{
    print "usage: perl prepare-config.perl <Config.sh> <build dir>\n";
    exit;
}

open(FILE, @ARGV[0]);
$path = @ARGV[1];


open(OUTFILE, ">${path}/galicconfig.h");
open(COUTF,   ">${path}/compile_time_info.c");

print COUTF "#include <stdio.h>\n";
print COUTF "void output_compile_time_options(void)\n\{\n";
print COUTF "printf(\n";

while($line=<FILE>)
{
    chop $line;

    @fields = split ' ' , $line;

    if(substr($fields[0], 0, 1) ne "#")
    {
	if(length($fields[0]) > 0)
	{
	    @subfields = split '=', $fields[0];

	    print OUTFILE "#define $subfields[0] $subfields[1]\n";
            print COUTF   "\"        $fields[0]\\n\"\n";
	}
    }
}

print COUTF "\"\\n\");\n";
print COUTF "\}\n";
