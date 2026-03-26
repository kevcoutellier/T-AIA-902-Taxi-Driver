import { useState, useRef, useEffect } from "react";

interface Props {
  title: string;
  children: React.ReactNode;
}

export default function InfoBubble({ title, children }: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const toggle = () => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.top - 4, left: r.right + 8 });
    }
    setOpen((v) => !v);
  };

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div className="info-bubble-wrap" ref={wrapRef}>
      <button className="info-btn" ref={btnRef} onClick={toggle} title={title}>
        ?
      </button>
      {open && (
        <div className="info-popup" style={{ top: pos.top, left: pos.left }}>
          <strong className="info-popup-title">{title}</strong>
          <div className="info-popup-body">{children}</div>
        </div>
      )}
    </div>
  );
}
