#!/bin/bash


LOG=${1:-/var/log/tomcat10/tomcat*.log}

grep -hE "SEVERE|ERROR|Exception|Caused by|ClassNotFound|NoClassDefFound|LifecycleException|FAIL" $LOG \
| grep -v "WebappClassLoaderBase loadClass" \
| grep -v "WebappClassLoaderBase findClass" \
| grep -v "FINER:" \
| grep -v "FINE:" \
| grep -v "FINEST:" \
| sed '/^[[:space:]]*$/d'
