import { Transition } from "framer-motion";

export const transitions: Record<string, Transition> = {
  enter: { type: "spring", stiffness: 260, damping: 28 },
  layout: { type: "spring", stiffness: 300, damping: 30 },
  micro: { type: "spring", stiffness: 400, damping: 25 },
  draw: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
};

export const stagger = {
  fast: 0.08,
  normal: 0.12,
};
