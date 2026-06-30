#!/bin/bash

read -p "continue to reset the rules?" CONT
if test $CONT = yes
then
	ufw reset
else 
	exit 1
fi
ufw default deny incoming
ufw default deny outgoing

# --- WireGuard tunnel establishment ---
ufw allow out on enp4s0 to 16.170.209.233 port 51820 proto udp

# replaces the two manual ssh rules
ufw allow ssh
ufw allow out proto tcp to any port 22

# --- HTTP/HTTPS outbound, internal networks only ---
ufw allow out proto tcp to 10.0.0.0/8 port 80
ufw allow out proto tcp to 10.0.0.0/8 port 443
ufw allow out proto tcp to 192.168.0.0/16 port 80
ufw allow out proto tcp to 192.168.0.0/16 port 443

# --- Tomcat (8080) and admin (8443) inbound/outbound, 192.x only ---
ufw allow in proto tcp from 192.168.0.0/16 to any port 8080
ufw allow in proto tcp from 192.168.0.0/16 to any port 8443
ufw allow out proto tcp to 192.168.0.0/16 port 8080
ufw allow out proto tcp to 192.168.0.0/16 port 8443

# --- Java debug (JDWP 5005) inbound/outbound, 192.x only ---
ufw allow in proto tcp from 192.168.0.0/16 to any port 5005
ufw allow out proto tcp to 192.168.0.0/16 port 5005

# --- ICMP ping out ---
#ufw allow out proto icmp

# --- ICMP ping in (rate limited) ---
#ufw limit in proto icmp

# --- Traceroute out (UDP 33434-33534, TTL-expired ICMP comes back in) ---
ufw allow out proto udp to any port 33434:33534

# --- DNS out (needed for basically everything) ---
ufw allow out proto udp to any port 53
ufw allow out proto tcp to any port 53

# --- Anti-flood: rate limit inbound on physical interface ---
ufw limit in on enp4s0 proto tcp
ufw limit in on enp4s0 proto udp

# --- WireGuard interface traffic ---
ufw allow in on wg0
ufw allow out on wg0

ufw enable
ufw status verbose
