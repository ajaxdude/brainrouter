//! Resolve the working directory of the process that owns a TCP peer socket.
//!
//! When OMP connects to brainrouter, each instance has a distinct cwd. By
//! mapping the peer's source port → client-side socket inode → owning pid →
//! /proc/{pid}/cwd we can surface the project folder in the dashboard without
//! any changes to OMP or its provider configuration.
//!
//! Only works on Linux (requires /proc/net/tcp and /proc/{pid}/fd).
//! Returns None silently on any I/O error so callers degrade gracefully.

use std::net::{IpAddr, SocketAddr};

/// Given the TCP peer address of an accepted connection (and the local
/// address of the accepting socket), return the cwd of the process that
/// owns the client-side socket, or None if it cannot be determined.
///
/// v2 L3 Step 9: the table scan MUST disambiguate by remote address, not
/// client port alone. Under test-suite (or busy-host) load, ephemeral ports
/// churn: a row whose local port equals the client port can belong to a
/// completely different process's LISTEN socket, and the inode of that
/// socket yields the wrong process's cwd (the source of the project-scope
/// flakes). Matching (local port, remote ip, remote port) identifies the
/// client socket uniquely.
pub fn peer_cwd(peer_addr: &SocketAddr, local_addr: &SocketAddr) -> Option<String> {
    let peer_port = peer_addr.port();
    let local_port = local_addr.port();

    let inode = match peer_addr.ip() {
        IpAddr::V4(_) => {
            find_inode_for_client_port(peer_port, local_port, "/proc/net/tcp")
        }
        IpAddr::V6(_) => {
            find_inode_for_client_port(peer_port, local_port, "/proc/net/tcp6")
        }
    }?;
    find_cwd_for_inode(inode)
}

/// Resolve the cwd of a process given its PID.
pub fn cwd_from_pid(pid: i32) -> Option<String> {
    let cwd_link = format!("/proc/{}/cwd", pid);
    if let Ok(cwd) = std::fs::read_link(&cwd_link) {
        return cwd.to_str().map(|s| s.trim_end_matches(" (deleted)").to_string());
    }
    None
}

/// Scan /proc/net/tcp (or tcp6) for the row where local_port == client_port
/// AND remote == the daemon's listening socket. Returns the socket inode.
fn find_inode_for_client_port(client_port: u16, local_port: u16, path: &str) -> Option<u64> {
    let data = std::fs::read_to_string(path).ok()?;
    find_inode_in_table(client_port, local_port, &data)
}

fn find_inode_in_table(client_port: u16, daemon_port: u16, data: &str) -> Option<u64> {
    for line in data.lines().skip(1) {
        // Fields are whitespace-separated; split_ascii_whitespace handles variable spacing.
        let mut cols = line.split_ascii_whitespace();
        let _sl = cols.next()?;
        let local_addr = cols.next()?; // "AABBCCDD:PPPP" (client socket)
        let rem_addr = cols.next()?; // "0100007F:DDDD" (the daemon socket)

        let row_local_port = local_addr
            .split(':')
            .nth(1)
            .and_then(|s| u16::from_str_radix(s, 16).ok())?;
        // Remote must be the daemon we connected to: the ONLY reliable
        // disambiguation under ephemeral-port churn (a bare port match can
        // land on a foreign LISTEN socket; its inode names the wrong
        // process, and the wrong cwd silently mis-scopes the request).
        // `rsplit(':')` yields the 4-hex-digit port for v4 ("0100007F:238B")
        // and v6 (32-hex:PPPP) rows alike.
        let row_remote_port = rem_addr.rsplit(':').next()?;
        let row_remote_port = u16::from_str_radix(row_remote_port, 16).ok()?;

        if row_local_port == client_port && row_remote_port == daemon_port {
            // Skip: st, tx_queue:rx_queue, tr:tm->when, retrnsmt, uid, timeout.
            for _ in 0..6 {
                cols.next()?;
            }
            let inode_str = cols.next()?;
            return inode_str.parse::<u64>().ok();
        }
    }
    None
}

/// Scan /proc/{pid}/fd for a symlink pointing to socket:[inode], then read
/// /proc/{pid}/cwd.
fn find_cwd_for_inode(inode: u64) -> Option<String> {
    let target = format!("socket:[{}]", inode);
    let proc = std::fs::read_dir("/proc").ok()?;

    for entry in proc.flatten() {
        let name = entry.file_name();
        let pid_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !pid_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        let fd_dir = format!("/proc/{}/fd", pid_str);
        let fds = match std::fs::read_dir(&fd_dir) {
            Ok(d) => d,
            Err(_) => continue, // no permission or process gone
        };

        for fd_entry in fds.flatten() {
            match std::fs::read_link(fd_entry.path()) {
                Ok(link) if link.to_str() == Some(&target) => {
                    // Found the process. Read its cwd.
                    let cwd_link = format!("/proc/{}/cwd", pid_str);
                    if let Ok(cwd) = std::fs::read_link(&cwd_link) {
                        return cwd.to_str().map(|s| s.trim_end_matches(" (deleted)").to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str =
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";

    #[test]
    fn parses_ipv4_socket_inode_column() {
        let table = format!(
            "{HEADER}   0: 0100007F:C350 0100007F:238B 01 00000000:00000000 02:00000000 00000000  1000 0 424242 2 0000000000000000\n"
        );
        assert_eq!(find_inode_in_table(0xC350, 0x238B, &table), Some(424242));
    }

    #[test]
    fn parses_ipv6_socket_inode_column() {
        // A connected (not wildcard) IPv6 row: remote == the daemon socket.
        let table = format!(
            "{HEADER}  12: 00000000000000000000000001000000:ABCD 00000000000000000000000001000000:1F90 01 00000000:00000000 00:00000000 00000000  1000 0 987654 1 0000000000000000\n"
        );
        assert_eq!(find_inode_in_table(0xABCD, 0x1F90, &table), Some(987654));
    }

    #[test]
    fn ignores_non_matching_ports() {
        let table = format!(
            "{HEADER}   0: 0100007F:C350 0100007F:238B 01 00000000:00000000 02:00000000 00000000  1000 0 424242\n"
        );
        assert_eq!(find_inode_in_table(0xC351, 0x238B, &table), None);
    }

    #[test]
    fn port_collision_needs_remote_match() {
        // v2 L3 Step 9: a foreign LISTEN socket reusing the client's
        // ephemeral port must NOT match; the real row (remote = daemon) does.
        let table = format!(
            "{HEADER}   0: 0100007F:4444 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000 0 111111 1 0000000000000000\n\
                1: 0100007F:4444 0100007F:1F90 01 00000000:00000000 02:00000000 00000000  1000 0 222222 2 0000000000000000\n"
        );
        assert_eq!(find_inode_in_table(0x4444, 0x1F90, &table), Some(222222));
    }
}
