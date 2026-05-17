/* ─────────────────────────────────────────────────────────────
   Smart Bill Scanner — OCR and Barcode Reader interface
   ───────────────────────────────────────────────────────────── */

import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../services/api';
import { useToast } from '../hooks/useToast';
import { 
  Camera, Upload, Sparkles, CheckCircle2, 
  Trash2, Plus, RefreshCw, Barcode, HelpCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function BillScanner() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const { success, error, warning, info } = useToast();

  // States
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState('');
  const [scanResult, setScanResult] = useState(null);
  const [barcode, setBarcode] = useState('');

  // Interactive Camera capture simulation
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);

  const startCamera = async () => {
    setCameraActive(true);
    setScanResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      info('Camera initialization active.');
    } catch (err) {
      console.error('Camera access failed:', err.message);
      setCameraActive(false);
      warning('Camera access denied or unavailable. Please upload a file instead.');
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    // Stop camera stream
    const stream = videoRef.current.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    videoRef.current.srcObject = null;
    setCameraActive(false);

    canvas.toBlob((blob) => {
      const capturedFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
      setFile(capturedFile);
      setPreview(URL.createObjectURL(capturedFile));
      success('Snapshot successfully captured!');
    }, 'image/jpeg');
  };

  const cancelCamera = () => {
    if (videoRef.current) {
      const stream = videoRef.current.srcObject;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setScanResult(null);
      success('Image file selected!');
    }
  };

  const triggerUpload = async () => {
    if (!file) return;
    setScanning(true);
    setProgress('Uploading receipt and initializing Tesseract OCR engine...');

    const formData = new FormData();
    formData.append('bill', file);
    if (barcode) formData.append('barcode', barcode);

    try {
      const response = await apiClient.post('/api/bills/scan', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setScanResult(response.data.billDetails);
      success('Receipt parsed successfully via AI OCR!');
    } catch (err) {
      const errMsg = err.message || 'OCR scanner parsing failed. Verify image clarity.';
      error(errMsg);
    } finally {
      setScanning(false);
      setProgress('');
    }
  };

  // Inline changes to parsed items
  const handleItemChange = (idx, field, value) => {
    const updated = [...scanResult.items];
    updated[idx] = { 
      ...updated[idx], 
      [field]: field === 'name' ? value : parseFloat(value) || 0 
    };
    // Recalculate total_price
    if (field === 'price' || field === 'quantity') {
      updated[idx].total_price = updated[idx].price * updated[idx].quantity;
    }
    
    // Recalculate bill total
    const itemSum = updated.reduce((sum, item) => sum + item.total_price, 0);
    const newTotal = itemSum + scanResult.tax_amount - scanResult.discount_amount;

    setScanResult({
      ...scanResult,
      total_amount: newTotal,
      items: updated
    });
  };

  const deleteItem = (idx) => {
    const updated = scanResult.items.filter((_, i) => i !== idx);
    const itemSum = updated.reduce((sum, item) => sum + item.total_price, 0);
    const newTotal = itemSum + scanResult.tax_amount - scanResult.discount_amount;

    setScanResult({
      ...scanResult,
      total_amount: newTotal,
      items: updated
    });
  };

  const addItem = () => {
    const newItem = {
      id: Math.random().toString(),
      name: 'New Item',
      quantity: 1,
      price: 0,
      total_price: 0,
      category_id: scanResult.items[0]?.category_id || '',
      category_name: 'Others'
    };
    setScanResult({
      ...scanResult,
      items: [...scanResult.items, newItem]
    });
  };

  const handleSaveBill = async () => {
    try {
      await apiClient.post('/api/bills/save', scanResult);
      success('Grocery bill logs committed to database!');
      navigate('/dashboard');
    } catch (err) {
      error(err.message || 'Database save failed.');
    }
  };

  return (
    <div className="space-y-8 pb-12 max-w-4xl mx-auto w-full">
      <div>
        <h2 className="text-2xl md:text-3xl font-extrabold text-white">Smart Bill Scanner</h2>
        <p className="text-sm text-gray-500 font-medium">Scan barcode, upload store receipt or capture using your web camera.</p>
      </div>

      <div className="grid md:grid-cols-12 gap-8">
        
        {/* Capture Panel */}
        <div className="md:col-span-5 space-y-6">
          <div className="glass-panel p-6 rounded-2xl flex flex-col justify-center items-center relative overflow-hidden bg-gray-950/40 min-h-[300px]">
            {cameraActive ? (
              <div className="w-full space-y-4">
                <video ref={videoRef} autoPlay playsInline className="w-full rounded-xl border border-gray-800 bg-black aspect-video object-cover" />
                <div className="flex gap-3">
                  <button onClick={capturePhoto} className="flex-1 py-3 bg-emerald-500 text-black font-extrabold rounded-xl text-xs hover:bg-emerald-400 transition">
                    Capture Photo
                  </button>
                  <button onClick={cancelCamera} className="px-4 py-3 bg-gray-900 border border-gray-800 text-gray-400 font-extrabold rounded-xl text-xs hover:text-white transition">
                    Cancel
                  </button>
                </div>
              </div>
            ) : preview ? (
              <div className="w-full space-y-4 text-center">
                <img src={preview} alt="Captured receipt preview" className="w-full rounded-xl border border-gray-800 max-h-[320px] object-contain bg-gray-950" />
                <div className="flex gap-2 justify-center">
                  <button 
                    onClick={() => { setFile(null); setPreview(''); setScanResult(null); }} 
                    className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20 rounded-xl transition"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                  <button 
                    onClick={startCamera} 
                    className="flex-1 py-3 px-4 bg-gray-900 border border-gray-800 text-white font-extrabold rounded-xl text-xs flex items-center justify-center gap-2 hover:border-emerald-500/30 transition"
                  >
                    <Camera className="h-4 w-4 text-emerald-400" />
                    Retake Photo
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-6 w-full text-center py-6">
                <div className="h-16 w-16 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto text-emerald-400">
                  <Camera className="h-8 w-8" />
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-white">Snap receipt or upload photo</p>
                  <p className="text-xs text-gray-500 leading-relaxed">Capture full receipt details including items and tax amount clearly.</p>
                </div>

                <div className="flex gap-3">
                  <button 
                    onClick={startCamera} 
                    className="flex-1 py-3 bg-emerald-500 text-black font-extrabold rounded-xl text-xs hover:bg-emerald-400 transition"
                  >
                    Use Camera
                  </button>
                  <button 
                    onClick={() => fileInputRef.current.click()} 
                    className="flex-1 py-3 bg-gray-900 border border-gray-800 text-white font-extrabold rounded-xl text-xs hover:border-emerald-500/30 transition"
                  >
                    Upload File
                  </button>
                </div>
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              </div>
            )}
          </div>

          {/* Barcode Option input */}
          <div className="glass-panel p-5 rounded-2xl space-y-3">
            <div className="flex items-center gap-2">
              <Barcode className="h-4.5 w-4.5 text-emerald-400" />
              <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Simulate Scan Barcode (Optional)</label>
            </div>
            <input 
              type="text" 
              value={barcode}
              onChange={(e) => setBarcode(e.target.value)}
              placeholder="e.g. 8901058002316" 
              className="w-full px-4 glass-input"
            />
            <p className="text-[10px] text-gray-500 leading-relaxed">
              If a product barcode is present, entering it will auto-populate product names from database.
            </p>
          </div>

          {/* Upload and trigger scanning */}
          {file && !scanning && !scanResult && (
            <button 
              onClick={triggerUpload}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 hover:opacity-90 text-black font-extrabold flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/10 transition"
            >
              <Sparkles className="h-5 w-5" />
              Run Intelligent OCR Scan
            </button>
          )}

          {scanning && (
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <div className="flex justify-between items-center text-xs font-bold">
                <span className="text-white flex items-center gap-2">
                  <RefreshCw className="h-4 w-4 text-emerald-400 animate-spin" />
                  Reading Invoices...
                </span>
              </div>
              <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500 rounded-full w-4/5 animate-pulse" />
              </div>
              <p className="text-[10px] text-gray-400 leading-relaxed text-center italic">{progress}</p>
            </div>
          )}
        </div>

        {/* Verification & Edit Ledger Panel */}
        <div className="md:col-span-7">
          <AnimatePresence>
            {scanResult ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="glass-panel p-6 rounded-3xl space-y-6 bg-gray-950/40"
              >
                <div className="flex justify-between items-center border-b border-gray-900 pb-4">
                  <div>
                    <h3 className="font-extrabold text-lg text-white">Extracted Receipt Logs</h3>
                    <p className="text-xs text-emerald-400 flex items-center gap-1 mt-0.5"><CheckCircle2 className="h-3.5 w-3.5" /> High confidence match</p>
                  </div>
                  <button onClick={addItem} className="p-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-xl transition">
                    <Plus className="h-5 w-5" />
                  </button>
                </div>

                {/* General parameters */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Store Name</label>
                    <input 
                      type="text" 
                      value={scanResult.store_name}
                      onChange={(e) => setScanResult({ ...scanResult, store_name: e.target.value })}
                      className="w-full px-4 glass-input py-2 text-xs" 
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Purchase Date</label>
                    <input 
                      type="date" 
                      value={scanResult.bill_date}
                      onChange={(e) => setScanResult({ ...scanResult, bill_date: e.target.value })}
                      className="w-full px-4 glass-input py-2 text-xs" 
                    />
                  </div>
                </div>

                {/* Items ledger */}
                <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
                  {scanResult.items.map((item, idx) => (
                    <div key={item.id} className="p-3.5 rounded-xl bg-gray-950 border border-gray-900 space-y-3 relative group">
                      <button 
                        onClick={() => deleteItem(idx)}
                        className="absolute right-3 top-3.5 opacity-0 group-hover:opacity-100 text-rose-500 hover:text-rose-400 transition"
                      >
                        <Trash2 className="h-4.5 w-4.5" />
                      </button>

                      <div className="grid grid-cols-12 gap-3 items-center">
                        {/* Name */}
                        <div className="col-span-7">
                          <input 
                            type="text" 
                            value={item.name}
                            onChange={(e) => handleItemChange(idx, 'name', e.target.value)}
                            className="w-full bg-transparent border-0 focus:ring-0 p-0 text-xs font-bold text-white placeholder-gray-600 focus:outline-none"
                            placeholder="Item Name"
                          />
                        </div>

                        {/* Qty */}
                        <div className="col-span-2">
                          <input 
                            type="number" 
                            value={item.quantity}
                            onChange={(e) => handleItemChange(idx, 'quantity', e.target.value)}
                            className="w-full bg-transparent border-0 focus:ring-0 p-0 text-xs font-bold text-emerald-400 text-center focus:outline-none"
                            placeholder="Qty"
                          />
                        </div>

                        {/* Total Price */}
                        <div className="col-span-3 text-right">
                          <input 
                            type="number" 
                            value={item.price}
                            onChange={(e) => handleItemChange(idx, 'price', e.target.value)}
                            className="w-full bg-transparent border-0 focus:ring-0 p-0 text-xs font-bold text-white text-right focus:outline-none"
                            placeholder="Price"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Totals */}
                <div className="border-t border-gray-900 pt-4 space-y-2.5 text-xs">
                  <div className="flex justify-between text-gray-500">
                    <span>GST Tax Levy</span>
                    <input 
                      type="number" 
                      value={scanResult.tax_amount}
                      onChange={(e) => {
                        const tax = parseFloat(e.target.value) || 0;
                        const itemSum = scanResult.items.reduce((s, it) => s + it.total_price, 0);
                        setScanResult({ ...scanResult, tax_amount: tax, total_amount: itemSum + tax - scanResult.discount_amount });
                      }}
                      className="w-20 bg-transparent border-0 p-0 text-right text-gray-400 font-bold focus:outline-none" 
                    />
                  </div>
                  <div className="flex justify-between text-gray-500">
                    <span>Discount Deducted</span>
                    <input 
                      type="number" 
                      value={scanResult.discount_amount}
                      onChange={(e) => {
                        const disc = parseFloat(e.target.value) || 0;
                        const itemSum = scanResult.items.reduce((s, it) => s + it.total_price, 0);
                        setScanResult({ ...scanResult, discount_amount: disc, total_amount: itemSum + scanResult.tax_amount - disc });
                      }}
                      className="w-20 bg-transparent border-0 p-0 text-right text-gray-400 font-bold focus:outline-none" 
                    />
                  </div>
                  <div className="flex justify-between font-extrabold text-sm text-white pt-2 border-t border-gray-900/60">
                    <span>Net Amount Due</span>
                    <span>₹{scanResult.total_amount.toFixed(2)}</span>
                  </div>
                </div>

                <button 
                  onClick={handleSaveBill}
                  className="w-full py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold shadow-lg shadow-emerald-500/10 transition"
                >
                  Confirm and Commit items
                </button>
              </motion.div>
            ) : (
              <div className="border-2 border-dashed border-gray-900 rounded-3xl h-full flex flex-col items-center justify-center text-center p-8 py-16">
                <HelpCircle className="h-10 w-10 text-gray-600 mb-4" />
                <h3 className="font-bold text-gray-400 mb-1 text-sm">Waiting for OCR processing</h3>
                <p className="text-xs text-gray-500 max-w-xs leading-relaxed">
                  Upload an image of your grocery receipt in the left panel to scan items.
                </p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
