"use client";

import { useEffect, useMemo, useState } from "react";
import { PIPELINE_STEPS } from "../lib/pipeline-steps";

interface Options {
  started: boolean;
  done: boolean;
  reducedMotion: boolean;
  onComplete?: () => void;
}

export function useInvestigationPipeline({
  started,
  done,
  reducedMotion,
  onComplete,
}: Options) {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (!started) {
      setActiveIndex(0);
      return;
    }

    if (reducedMotion) {
      if (done) {
        setActiveIndex(PIPELINE_STEPS.length);
        onComplete?.();
      } else {
        setActiveIndex(1);
      }
      return;
    }

    if (done) {
      const interval = window.setInterval(() => {
        setActiveIndex((prev) => {
          const next = Math.min(PIPELINE_STEPS.length, prev + 1);
          if (next >= PIPELINE_STEPS.length) {
            window.clearInterval(interval);
            onComplete?.();
          }
          return next;
        });
      }, 80);
      return () => window.clearInterval(interval);
    }

    let cancelled = false;
    const run = async () => {
      for (let i = 0; i < PIPELINE_STEPS.length; i += 1) {
        if (cancelled || done) return;
        setActiveIndex(i + 1);
        await new Promise((resolve) =>
          window.setTimeout(resolve, PIPELINE_STEPS[i].dwellMs)
        );
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [started, done, reducedMotion, onComplete]);

  return useMemo(() => {
    return PIPELINE_STEPS.map((step, index) => ({
      ...step,
      completed: done
        ? index < activeIndex
        : index < activeIndex - 1,
      active:
        !done &&
        index === activeIndex - 1 &&
        activeIndex <= PIPELINE_STEPS.length,
    }));
  }, [activeIndex, done]);
}
