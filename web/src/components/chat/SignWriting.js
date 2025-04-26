import React, {useEffect, useRef, useState} from "react";
import axios from "axios";
import Cookies from "js-cookie";

const SignWriting = ({showCamera, setShowCamera, setNewMessage, username, answer}) => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [videoBlob, setVideoBlob] = useState(null);
  const [playbackUrl, setPlaybackUrl] = useState(null);
  const[uploading, setUploading] = useState(false);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });

    videoRef.current.srcObject = stream;

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;

    const chunks = [];
    mediaRecorder.ondataavailable = (event) => chunks.push(event.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      setVideoBlob(blob);
      setPlaybackUrl(URL.createObjectURL(blob));

      videoRef.current.srcObject = null;
      videoRef.current.src = URL.createObjectURL(blob);
      videoRef.current.controls = true;
    };

    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
    setRecording(false);
  };

  const uploadVideo = async () => {
    if (!videoBlob) return;

    setUploading(true);

    const formData = new FormData();
    formData.append("videoFile", videoBlob, "sign_language_video.webm");
    formData.append("username", username);
    formData.append("answer", answer);
    console.log("Username is "+username+" and Answer is "+answer);
    try {
      const accessToken = Cookies.get('accessToken');
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/v1/threads/sign/sign_writing`,
          formData, {
            headers: {
              Authorization: `Bearer ${accessToken}`,
              "Content-Type": "multipart/form-data"
            },
          });

      const text = response.data;
      setNewMessage(text);
      if (showCamera) {
        setShowCamera(false);
      }
    }catch(error){
      console.error("Upload failed:", error);
    }finally{
      setUploading(false);
    }
  };

  return (
    <div>
      <h3>Sign Language Recording</h3>

      <video ref={videoRef} autoPlay muted width="400" height="300"></video>

      <br />

      {!recording ? (
        <button onClick={startRecording}>🎥 Start Recording</button>
      ) : (
        <button onClick={stopRecording}>⏹ Stop Recording</button>
      )}

      <button onClick={uploadVideo} disabled={!videoBlob || uploading}>
        📤 Upload Video
      </button>
    </div>
  );
};

export default SignWriting;
