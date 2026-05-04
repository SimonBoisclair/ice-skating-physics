import React, { createContext, useContext, useReducer, useCallback } from 'react';

/** Default slider / input values. */
const DEFAULT_STATE = {
  gx: 0,
  gy: 0.15,
  gz: 0.9,
  gvx: 5,
  gvy: 0,
  gvz: 0,
  alpha: 0,
  lean: 15,
  pitch: 0,
  px: 0,
  py: 0,
  mass: 85,
  temp: -5,
  hollow: 15.875,
};

/** GPU-side state received over WebSocket. */
const DEFAULT_GPU = {};

const PhysicsContext = createContext(null);

function reducer(state, action) {
  switch (action.type) {
    case 'SET_PARAM':
      return { ...state, params: { ...state.params, [action.key]: action.value } };
    case 'SET_GPU':
      return { ...state, gpu: action.payload };
    default:
      return state;
  }
}

export function PhysicsProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, {
    params: DEFAULT_STATE,
    gpu: DEFAULT_GPU,
  });

  const setParam = useCallback((key, value) => {
    dispatch({ type: 'SET_PARAM', key, value: parseFloat(value) });
  }, []);

  const setGpu = useCallback((payload) => {
    dispatch({ type: 'SET_GPU', payload });
  }, []);

  return (
    <PhysicsContext.Provider value={{ params: state.params, gpu: state.gpu, setParam, setGpu }}>
      {children}
    </PhysicsContext.Provider>
  );
}

export function usePhysics() {
  const ctx = useContext(PhysicsContext);
  if (!ctx) throw new Error('usePhysics must be used within PhysicsProvider');
  return ctx;
}
