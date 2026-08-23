"use client";

import { useReducedMotion } from "framer-motion";
import { useInvestigationPipeline } from "../../hooks/useInvestigationPipeline";
import { FadeUp } from "../motion/FadeUp";
import { TimelineStep } from "./TimelineStep";
import styles from "./Investigation.module.css";

interface InvestigationViewProps {
  started: boolean;
  done: boolean;
  onComplete: () => void;
  onCancel: () => void;
}

export function InvestigationView({
  started,
  done,
  onComplete,
  onCancel,
}: InvestigationViewProps) {
  const reducedMotion = useReducedMotion();
  const steps = useInvestigationPipeline({
    started,
    done,
    reducedMotion: Boolean(reducedMotion),
    onComplete,
  });

  return (
    <FadeUp className={styles.investigation}>
      <h2 className={styles.heading}>Investigation in progress</h2>
      <div className={styles.timeline}>
        {steps.map((step, index) => (
          <TimelineStep
            key={step.id}
            label={step.label}
            active={step.active}
            completed={step.completed}
            isLast={index === steps.length - 1}
          />
        ))}
      </div>
      <button type="button" className={styles.cancel} onClick={onCancel}>
        Cancel
      </button>
    </FadeUp>
  );
}
