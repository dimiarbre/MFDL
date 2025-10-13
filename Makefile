# Rule to match all .def files
DEFFILES := $(wildcard *.def)

%.sif: %.def requirement.txt
	export SINGULARITY_TMPDIR=~/.tmp; singularity build $@ $<

all-def: container.sif

clean:
	rm -f *.sif