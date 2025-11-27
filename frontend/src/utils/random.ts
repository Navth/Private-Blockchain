export const randomFromArray = <T>(items: T[]): T => {
  const index = Math.floor(Math.random() * items.length);
  return items[index];
};

export const randomNumber = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const randomHex = (bytes: number): string => {
  const buffer = new Uint8Array(bytes);
  const cryptoApi = globalThis.crypto;

  if (cryptoApi?.getRandomValues) {
    cryptoApi.getRandomValues(buffer);
  } else {
    for (let i = 0; i < bytes; i += 1) {
      buffer[i] = Math.floor(Math.random() * 256);
    }
  }

  return Array.from(buffer)
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
};

export const randomSignature = (length = 16): string => {
  return randomHex(length).slice(0, length);
};
