#!/bin/bash
for i in *
do 
	if test -f "$i" -a \! -x "$i" && ! expr "$f" : .*.sh >/dev/null
	then
		echo $i
		unlink "$i"
	fi
done
