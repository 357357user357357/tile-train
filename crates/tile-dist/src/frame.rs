//! Length-prefixed frame IO over TCP.
//!
//! Frame = 4-byte little-endian payload length + payload bytes. Bounded to
//! 512 MiB to turn a corrupted length prefix into an error, not an OOM.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::TcpStream;

const MAX_FRAME: usize = 512 << 20;

/// Send one frame: blocking, whole payload.
pub fn send_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<()> {
    let len = u32::try_from(payload.len()).context("frame too large for u32")?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}

/// Receive one frame: blocking, exact length.
pub fn recv_frame(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .context("reading frame length (peer closed?)")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME {
        bail!("frame length {len} exceeds {MAX_FRAME}");
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

/// Send a JSON-serializable control message as one frame.
pub fn send_msg<T: serde::Serialize>(stream: &mut TcpStream, msg: &T) -> Result<()> {
    let bytes = serde_json::to_vec(msg)?;
    send_frame(stream, &bytes)
}

/// Receive one frame and deserialize it as a control message.
pub fn recv_msg<T: serde::de::DeserializeOwned>(stream: &mut TcpStream) -> Result<T> {
    let bytes = recv_frame(stream)?;
    serde_json::from_slice(&bytes).context("deserializing control message")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn frame_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let msg = crate::proto::ControlMsg::Loss {
                rank: 2,
                step: 99,
                loss: 0.125,
            };
            send_msg(&mut sock, &msg).unwrap();
            let back: crate::proto::ControlMsg = recv_msg(&mut sock).unwrap();
            assert_eq!(back, msg);
        });
        let mut sock = TcpStream::connect(addr).unwrap();
        let back: crate::proto::ControlMsg = recv_msg(&mut sock).unwrap();
        assert_eq!(back, crate::proto::ControlMsg::Loss { rank: 2, step: 99, loss: 0.125 });
        send_msg(&mut sock, &back).unwrap();
        handle.join().unwrap();
    }
}
