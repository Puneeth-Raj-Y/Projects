/* ─────────────────────────────────────────────────────────────
   Custom Hook — useToast
   ───────────────────────────────────────────────────────────── */

import { useToast as useToastContext } from '../context/ToastContext';

export function useToast() {
  const { addToast, removeToast } = useToastContext();

  return {
    toast: addToast,
    success: (msg, dur) => addToast(msg, 'success', dur),
    error: (msg, dur) => addToast(msg, 'error', dur),
    warning: (msg, dur) => addToast(msg, 'warning', dur),
    info: (msg, dur) => addToast(msg, 'info', dur),
    dismiss: removeToast,
  };
}

export default useToast;
