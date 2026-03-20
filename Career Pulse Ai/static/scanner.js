document.addEventListener('DOMContentLoaded', () => {
    const scanBtn = document.getElementById('scan-btn');
    const qrReader = document.getElementById('qr-reader');
    let html5QrCode;

    scanBtn.addEventListener('click', () => {
        if (qrReader.style.display === 'none') {
            qrReader.style.display = 'block';
            scanBtn.innerText = '🛑 Stop Scanner';
            startScanner();
        } else {
            qrReader.style.display = 'none';
            scanBtn.innerText = '📷 Scan Job QR';
            if (html5QrCode) html5QrCode.stop();
        }
    });

    function startScanner() {
        html5QrCode = new Html5Qrcode("qr-reader");
        const config = { fps: 10, qrbox: { width: 250, height: 250 } };

        html5QrCode.start(
            { facingMode: "environment" }, 
            config,
            (decodedText) => {
                // Handle the decoded QR code
                console.log(`Code matched = ${decodedText}`);
                alert(`QR Scanned: ${decodedText}\nRedirecting to AI matching...`);
                
                // If it's a URL, navigate to it or process it
                if (decodedText.startsWith('http')) {
                    window.location.href = decodedText;
                } else {
                    // Search for the scanned text
                    const roleInput = document.querySelector('input[name="role"]');
                    roleInput.value = decodedText;
                    document.querySelector('form').submit();
                }
                
                html5QrCode.stop();
                qrReader.style.display = 'none';
            },
            (errorMessage) => {
                // parse error, ignore it.
            }
        ).catch((err) => {
            console.error(`Unable to start scanning: ${err}`);
        });
    }
});
